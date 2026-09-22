"""Hierarchical universal document interpretation prototype for Sprint G3.

This module is isolated from production ingestion.  It prototypes a scalable
two-phase interpretation architecture:

CanonicalDocumentInput -> global reconnaissance -> local region analysis ->
deterministic reconciliation -> G2-compatible DocumentStructure.
"""

from __future__ import annotations

import base64
import difflib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from canonical_document import CanonicalBlock, CanonicalDocumentInput, SourcePosition
from canonical_document import ROLE_BODY, ROLE_HEADING, ROLE_LIST, ROLE_TABLE, ROLE_TITLE
from universal_structure_interpreter import (
    CONFIDENCE_VALUES,
    MAX_STRUCTURED_INPUT_CHARS,
    SECTION_SCOPE_VALUES,
    DocumentSection,
    DocumentStructure,
    ResponseDiagnostics,
    UniversalStructureError,
    confidence_rank,
    extract_response_diagnostics,
    parse_model_response,
    response_text,
)
from universal_structure_mapping import (
    STATUS_ACCEPTED,
    STATUS_FALLBACK,
    STATUS_PARTIAL,
    DocumentStructureMappingResult,
    build_mapping_result,
)


ANALYZER_NAME = "hierarchical_universal_model_structure"
ANALYZER_VERSION = "g3-prototype"

STRUCTURE_STATUS_EXCELLENT = "excellent"
STRUCTURE_STATUS_USABLE = "usable"
STRUCTURE_STATUS_WEAK = "weak"

MAPPING_STATUS_HIGH = "high"
MAPPING_STATUS_PARTIAL = "partial"
MAPPING_STATUS_LOW = "low"

STRUCTURAL_ROLE_MACRO = "macro_section"
STRUCTURAL_ROLE_SECTION = "section"
STRUCTURAL_ROLE_SUBSECTION = "subsection"
STRUCTURAL_ROLE_CONTINUATION = "continuation"
STRUCTURAL_ROLE_LOCAL = "local_structure"
STRUCTURAL_ROLES = (
    STRUCTURAL_ROLE_MACRO,
    STRUCTURAL_ROLE_SECTION,
    STRUCTURAL_ROLE_SUBSECTION,
    STRUCTURAL_ROLE_CONTINUATION,
    STRUCTURAL_ROLE_LOCAL,
)
SEGMENTING_ROLES = {
    STRUCTURAL_ROLE_MACRO,
    STRUCTURAL_ROLE_SECTION,
    STRUCTURAL_ROLE_SUBSECTION,
}

LOCAL_CLASS_STRUCTURAL = "structural_heading"
LOCAL_CLASS_CONTINUATION = "continuation_heading"
LOCAL_CLASS_LOCAL = "local_information_organization"
LOCAL_CLASS_NON_STRUCTURAL = "non_structural_content"
LOCAL_CLASSIFICATIONS = (
    LOCAL_CLASS_STRUCTURAL,
    LOCAL_CLASS_CONTINUATION,
    LOCAL_CLASS_LOCAL,
    LOCAL_CLASS_NON_STRUCTURAL,
)

MAX_REGION_TEXT_CHARS = 55_000
MAX_LOCAL_ANALYSIS_BLOCKS = 8
MAX_LOCAL_ANALYSIS_CHARS = 16_000
SMALL_DOCUMENT_BLOCK_LIMIT = 18
WINDOW_TARGET_BLOCKS = MAX_LOCAL_ANALYSIS_BLOCKS
WINDOW_OVERLAP_BLOCKS = 2
SUBSCOPE_OVERLAP_BLOCKS = 0
MAX_CANDIDATES_PER_LOCAL_CALL = 80
MAX_CANDIDATE_CONTEXT_CHARS = 18_000
MAX_HIERARCHICAL_MODEL_CALLS = 32


def _source_position_schema() -> Dict[str, Any]:
    return {
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
    }


GLOBAL_RECONNAISSANCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "document_title",
        "document_type",
        "structure_confidence",
        "organization_style",
        "macro_regions",
        "notes",
    ],
    "properties": {
        "document_title": {"type": "string"},
        "document_type": {"type": "string"},
        "structure_confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        "organization_style": {"type": "string"},
        "macro_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "semantic_title",
                    "source_title",
                    "approximate_start",
                    "approximate_end",
                    "confidence",
                    "evidence_types",
                ],
                "properties": {
                    "semantic_title": {"type": "string"},
                    "source_title": {"type": ["string", "null"]},
                    "approximate_start": _source_position_schema(),
                    "approximate_end": _source_position_schema(),
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
                    "evidence_types": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "notes": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
    },
}


LOCAL_STRUCTURE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "region_confidence",
        "sections",
    ],
    "properties": {
        "region_confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "group_id",
                    "source_title",
                    "level",
                    "parent",
                    "expected_start",
                    "expected_end",
                    "confidence",
                    "structural_role",
                ],
                "properties": {
                    "candidate_id": {"type": ["string", "null"]},
                    "group_id": {"type": ["string", "null"]},
                    "source_title": {"type": ["string", "null"]},
                    "level": {"type": "integer", "minimum": 1, "maximum": 8},
                    "parent": {"type": ["string", "null"]},
                    "expected_start": _source_position_schema(),
                    "expected_end": _source_position_schema(),
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
                    "structural_role": {
                        "type": "string",
                        "enum": [
                            STRUCTURAL_ROLE_SECTION,
                            STRUCTURAL_ROLE_SUBSECTION,
                            STRUCTURAL_ROLE_LOCAL,
                        ],
                    },
                },
            },
        },
    },
}


GLOBAL_RECON_INSTRUCTIONS = """\
You are an expert academic document-structure analyst.

Perform GLOBAL RECONNAISSANCE only. Identify the broad semantic organization
of the document. Do not enumerate every heading. Do not list ordinary body
content. Local bullets, examples, diagram labels, repeated presentation titles,
individual list items, captions, formulas, page artifacts, and tables are not
macro-regions unless they genuinely define a major document region.

Native style/role/slide/bullet/heading metadata is evidence, not truth.
Use source content and source order to infer major regions. Keep the response
compact and document-agnostic.

Distinguish semantic_title from source_title carefully:
- semantic_title is your normalized description of the conceptual region.
- source_title is literal or near-literal heading text visibly present in the
  source.
- Prefer exact source wording for source_title.
- Do not put semantic paraphrases in source_title.
- If no literal source heading is confidently visible, set source_title to null.
"""


LOCAL_ANALYSIS_INSTRUCTIONS = """\
You are an expert academic document-structure analyst.

Analyze ONLY the bounded source scope provided.

Your task is compact PHYSICAL STRUCTURE extraction: identify literal source
boundaries that the document itself uses as genuine sections or subsections.
Do not summarize the scope. Do not redesign the document. Do not group content
by conceptual similarity.

A structural boundary is something such as a chapter heading, section heading,
subsection heading, or genuine titled division of the material.

Do NOT promote ordinary content to hierarchy, including bullets, list entries,
sentences, definitions, examples, figure labels, diagram labels, table cells,
isolated emphasized words, repeated page titles, continuation text, or ordinary
paragraph lead-ins, unless the source clearly uses that item as an actual
section/subsection heading.

source_title must be literal text visible in the supplied source scope. Prefer
exact source wording. If no literal heading is present, do not invent one.

Preserve physical source order. Do not emit semantic explanations,
descriptions, summaries, reasons, or local-content inventories.

Native metadata is evidence, not truth. Return only structural boundaries
supported by the supplied source scope.

If a candidate_group_proposal represents multiple physical candidates that
form ONE logical heading, emit one section using that group_id. Do not emit the
group's marker/subtitle components separately. If the proposal is merely an
ambiguous stack or separate hierarchy, emit the relevant candidate_id values
individually instead.
"""


GLOBAL_RECON_TASK = """\
Return compact global reconnaissance for this document: title, type,
organization_style, and meaningful macro_regions with approximate source
locators. Do not exhaustively enumerate headings.
"""


LOCAL_ANALYSIS_TASK = """\
Return only genuine local structural boundaries for this bounded source scope.
Use literal source_title text, compact hierarchy fields, and physical source
positions. Do not describe ordinary body content.
"""


@dataclass(frozen=True)
class MacroRegion:
    region_id: str
    semantic_title: str
    source_title: Optional[str]
    approximate_start: Optional[SourcePosition]
    approximate_end: Optional[SourcePosition]
    confidence: str
    evidence_types: List[str]
    source_order: int
    source_start_order: Optional[int] = None
    source_end_order: Optional[int] = None


@dataclass(frozen=True)
class MacroRegionGrounding:
    region_id: str
    source_title: Optional[str]
    semantic_title: str
    source_position: Optional[SourcePosition]
    block_list_index: Optional[int]
    method: str
    confidence: float
    fidelity: str
    status: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class LocalAnalysisScope:
    scope_id: str
    scope_type: str
    region: MacroRegion
    blocks: List[CanonicalBlock]
    start_unit: str
    end_unit: str
    block_count: int
    input_char_count: int
    skipped: bool = False
    skip_reason: Optional[str] = None
    macro_region_id: Optional[str] = None
    subscope_index: int = 1
    subdivision_reason: str = "none"
    source_start: Optional[SourcePosition] = None
    source_end: Optional[SourcePosition] = None
    candidates: List["StructuralCandidate"] = field(default_factory=list)
    candidate_groups: List["CandidateGroupProposal"] = field(default_factory=list)
    call_budget_exhausted: bool = False


@dataclass(frozen=True)
class StructuralCandidate:
    candidate_id: str
    source_title: str
    source_position: SourcePosition
    source_order: int
    unit_type: str
    unit_index: int
    block_index: int
    start_offset: int
    end_offset: Optional[int]
    role_hint: str
    style_hint: Optional[str]
    native_hierarchy_hint: Optional[int]
    text_length: int
    numbering_pattern: Optional[str]
    evidence_signals: List[str]
    candidate_score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    macro_region_id: Optional[str] = None
    physical_context: Optional["CandidatePhysicalContext"] = None


@dataclass(frozen=True)
class CandidatePhysicalContext:
    previous_candidate_id: Optional[str]
    next_candidate_id: Optional[str]
    previous_block_distance: Optional[int]
    next_block_distance: Optional[int]
    same_unit_as_previous: Optional[bool]
    same_unit_as_next: Optional[bool]
    adjacent_to_previous: bool
    adjacent_to_next: bool
    body_content_before_previous_candidate: Optional[bool]
    body_content_before_next_candidate: Optional[bool]
    compatible_with_previous: Optional[bool]
    compatible_with_next: Optional[bool]


@dataclass(frozen=True)
class CandidateGroupProposal:
    group_id: str
    candidate_ids: List[str]
    source_start: SourcePosition
    source_end: SourcePosition
    relationship_hypothesis: str
    confidence: str
    evidence_signals: List[str]
    primary_candidate_id: str
    rejected: bool = False
    rejection_reason: Optional[str] = None
    macro_region_id: Optional[str] = None


@dataclass(frozen=True)
class LogicalStructuralNode:
    node_id: str
    candidate_ids: List[str]
    primary_candidate_id: str
    source_title: Optional[str]
    display_title: Optional[str]
    level: int
    parent_title: Optional[str]
    expected_start: Optional[SourcePosition]
    expected_end: Optional[SourcePosition]
    confidence: str
    structural_role: str
    source_order: int
    region_id: Optional[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MacroBoundaryReconciliation:
    reconnaissance: DocumentReconnaissance
    groundings: List[MacroRegionGrounding]
    candidates: List[StructuralCandidate]
    candidate_groups: List[CandidateGroupProposal]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class DocumentReconnaissance:
    document_title: str
    document_type: str
    structure_confidence: str
    organization_style: str
    macro_regions: List[MacroRegion]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LocalSection:
    candidate_id: Optional[str]
    group_id: Optional[str]
    component_candidate_ids: List[str]
    source_title: Optional[str]
    semantic_title: str
    level: int
    parent_title: Optional[str]
    expected_start: Optional[SourcePosition]
    expected_end: Optional[SourcePosition]
    confidence: str
    structural_role: str
    classification: str
    source_order: int


@dataclass(frozen=True)
class LocalObservation:
    title: str
    expected_start: Optional[SourcePosition]
    confidence: str
    kind: Optional[str] = None


@dataclass(frozen=True)
class LocalRegionStructure:
    region_id: str
    region_title: str
    region_confidence: str
    sections: List[LocalSection]
    continuation_nodes: List[LocalObservation] = field(default_factory=list)
    local_only_nodes: List[LocalObservation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    interpretation_failed: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalSection:
    section_id: str
    source_title: Optional[str]
    semantic_title: str
    level: int
    parent_id: Optional[str]
    parent_title: Optional[str]
    region_id: Optional[str]
    expected_start: Optional[SourcePosition]
    expected_end: Optional[SourcePosition]
    confidence: str
    structural_role: str
    source_order: int


@dataclass(frozen=True)
class StructureQuality:
    score: float
    status: str
    reasons: List[str]


@dataclass(frozen=True)
class MappingFidelity:
    score: float
    status: str
    reasons: List[str]


@dataclass(frozen=True)
class CanonicalDocumentStructure:
    document_title: str
    document_type: str
    structure_quality: StructureQuality
    macro_regions: List[MacroRegion]
    sections: List[CanonicalSection]
    diagnostics: Dict[str, Any]

    def to_g2_document_structure(self) -> DocumentStructure:
        id_map = {section.section_id: section for section in self.sections}
        return DocumentStructure(
            document_title=self.document_title,
            document_type=self.document_type,
            confidence=_confidence_from_quality(self.structure_quality),
            analyzer=ANALYZER_NAME,
            analyzer_version=ANALYZER_VERSION,
            sections=[
                DocumentSection(
                    section_id=section.section_id,
                    source_title=section.source_title,
                    semantic_title=section.semantic_title,
                    level=section.level,
                    parent_id=section.parent_id,
                    parent_title=(
                        id_map[section.parent_id].source_title
                        or id_map[section.parent_id].semantic_title
                    )
                    if section.parent_id in id_map
                    else section.parent_title,
                    source_order=section.source_order,
                    expected_start=section.expected_start,
                    expected_end=section.expected_end,
                    confidence=section.confidence,
                    scope="global",
                )
                for section in self.sections
                if section.structural_role in SEGMENTING_ROLES
            ],
            local_structure_summary={
                "continuation_nodes": self.diagnostics.get("continuation_nodes", 0),
                "local_only_nodes": self.diagnostics.get("local_only_nodes", 0),
            },
            notes=list(self.diagnostics.get("notes", [])[:5]),
        )


@dataclass(frozen=True)
class HierarchicalInterpretationResult:
    reconnaissance: DocumentReconnaissance
    groundings: List[MacroRegionGrounding]
    local_scopes: List[LocalAnalysisScope]
    local_structures: List[LocalRegionStructure]
    canonical_structure: CanonicalDocumentStructure
    mapping_result: DocumentStructureMappingResult
    mapping_fidelity: MappingFidelity
    diagnostics: Dict[str, Any]
    elapsed_seconds: float


def build_text_config(schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }


def build_document_input(
    task: str,
    document: CanonicalDocumentInput,
    *,
    original_file_bytes: Optional[bytes] = None,
    blocks: Optional[Sequence[CanonicalBlock]] = None,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": task}]
    if context:
        content.append({"type": "input_text", "text": context})
    if document.file_type == "pdf" and original_file_bytes and blocks is None:
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
                "text": build_structured_text(document, blocks=blocks),
            }
        )
    return [{"role": "user", "content": content}]


def build_structured_text(
    document: CanonicalDocumentInput,
    *,
    blocks: Optional[Sequence[CanonicalBlock]] = None,
    max_chars: int = MAX_STRUCTURED_INPUT_CHARS,
) -> str:
    chosen_blocks = list(blocks) if blocks is not None else document.blocks
    lines = [
        f"FILE_TYPE: {document.file_type}",
        f"FILENAME: {document.filename}",
        f"TITLE: {document.title}",
        "",
    ]
    current_unit = None
    for block in chosen_blocks:
        unit_key = (block.source_position.unit_type, block.source_position.unit_index)
        if unit_key != current_unit:
            current_unit = unit_key
            lines.append(
                f"[unit={block.source_position.unit_type}:{block.source_position.unit_index}]"
            )
        hint_parts = [
            f"block={block.source_position.block_index}",
            f"order={block.source_order}",
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
    if len(text) > max_chars:
        return text[:max_chars] + "\n[TRUNCATED_FOR_G3_PROTOTYPE]"
    return text


def interpret_hierarchical_structure(
    *,
    document: CanonicalDocumentInput,
    model: str,
    client: Optional[Any] = None,
    original_file_bytes: Optional[bytes] = None,
    max_output_tokens: int = 6000,
) -> HierarchicalInterpretationResult:
    if client is None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise UniversalStructureError("OpenAI SDK is unavailable.") from exc
        client = OpenAI()

    started = time.perf_counter()
    reconnaissance, recon_diag = run_global_reconnaissance(
        document=document,
        model=model,
        client=client,
        original_file_bytes=original_file_bytes,
        max_output_tokens=max_output_tokens,
    )
    groundings = ground_macro_regions(document, reconnaissance)
    candidates = discover_structural_candidates(document, reconnaissance, groundings)
    candidate_groups = propose_candidate_groups(document, candidates)
    macro_reconciliation = reconcile_macro_boundaries(
        document,
        reconnaissance,
        groundings,
        candidates,
        candidate_groups,
    )
    reconnaissance = macro_reconciliation.reconnaissance
    groundings = macro_reconciliation.groundings
    candidates = macro_reconciliation.candidates
    candidate_groups = macro_reconciliation.candidate_groups
    scopes = prepare_candidate_analysis_scopes(
        document,
        reconnaissance,
        groundings,
        candidates,
        candidate_groups,
        max_model_calls=MAX_HIERARCHICAL_MODEL_CALLS - 1,
    )
    local_structures = []
    local_diagnostics = []
    for scope in scopes:
        if scope.skipped:
            local = LocalRegionStructure(
                region_id=scope.region.region_id,
                region_title=scope.region.semantic_title,
                region_confidence="LOW",
                sections=[],
                notes=[scope.skip_reason or "local analysis skipped"],
                interpretation_failed=True,
                diagnostics=local_scope_diagnostics(scope),
            )
            local_structures.append(local)
            local_diagnostics.append(local.diagnostics)
            continue
        local, diagnostics = run_local_structure_analysis(
            document=document,
            scope=scope,
            model=model,
            client=client,
            max_output_tokens=max_output_tokens,
        )
        local_structures.append(local)
        local_diagnostics.append(diagnostics)

    canonical = reconcile_hierarchy(document, reconnaissance, local_structures)
    mapping_result = build_mapping_result(document, canonical.to_g2_document_structure())
    mapping_fidelity = compute_mapping_fidelity(mapping_result)
    elapsed = time.perf_counter() - started
    return HierarchicalInterpretationResult(
        reconnaissance=reconnaissance,
        groundings=groundings,
        local_scopes=scopes,
        local_structures=local_structures,
        canonical_structure=canonical,
        mapping_result=mapping_result,
        mapping_fidelity=mapping_fidelity,
        diagnostics={
            "model_calls": 1 + sum(1 for local in local_structures if not (local.diagnostics or {}).get("skipped")),
            "global_diagnostics": recon_diag,
            "local_diagnostics": local_diagnostics,
            "candidate_diagnostics": candidate_discovery_diagnostics(document, candidates, scopes),
            "macro_boundary_reconciliation": macro_reconciliation.diagnostics,
            "candidate_grouping_diagnostics": candidate_grouping_diagnostics(candidate_groups),
        },
        elapsed_seconds=elapsed,
    )


def run_global_reconnaissance(
    *,
    document: CanonicalDocumentInput,
    model: str,
    client: Any,
    original_file_bytes: Optional[bytes] = None,
    max_output_tokens: int = 6000,
) -> Tuple[DocumentReconnaissance, ResponseDiagnostics]:
    response = client.responses.create(
        model=model,
        instructions=GLOBAL_RECON_INSTRUCTIONS,
        input=build_document_input(
            GLOBAL_RECON_TASK,
            document,
            original_file_bytes=original_file_bytes,
        ),
        text=build_text_config(
            GLOBAL_RECONNAISSANCE_SCHEMA,
            "douno_hierarchical_global_reconnaissance_g3",
        ),
        max_output_tokens=max_output_tokens,
        store=False,
    )
    raw_text = response_text(response)
    diagnostics = extract_response_diagnostics(response, raw_text)
    parsed = parse_model_response(raw_text, diagnostics)
    return reconnaissance_from_json(parsed), diagnostics


def run_local_structure_analysis(
    *,
    document: CanonicalDocumentInput,
    scope: LocalAnalysisScope,
    model: str,
    client: Any,
    max_output_tokens: int = 6000,
) -> Tuple[LocalRegionStructure, ResponseDiagnostics]:
    context = (
        f"SCOPE_ID: {scope.scope_id}\n"
        f"SCOPE_TYPE: {scope.scope_type}\n"
        f"MACRO_REGION_ID: {scope.region.region_id}\n"
        f"MACRO_REGION_TITLE: {scope.region.source_title or scope.region.semantic_title}\n"
        f"MACRO_REGION_CONFIDENCE: {scope.region.confidence}\n"
        f"SOURCE_UNITS: {scope.start_unit} -> {scope.end_unit}\n"
        f"BLOCKS: {scope.block_count}\n"
        f"INPUT_CHARS: {scope.input_char_count}\n"
        f"CANDIDATES: {len(scope.candidates)}\n"
        f"CANDIDATE_GROUP_PROPOSALS: {len(scope.candidate_groups)}"
    )
    if scope.candidates:
        context = context + "\n\n" + build_candidate_context(scope)
    started = time.perf_counter()
    raw_text = ""
    diagnostics: Optional[ResponseDiagnostics] = None
    failure_reason = None
    try:
        response = client.responses.create(
            model=model,
            instructions=LOCAL_ANALYSIS_INSTRUCTIONS,
            input=build_document_input(
                LOCAL_ANALYSIS_TASK,
                document,
                blocks=[] if scope.candidates else scope.blocks,
                context=context,
            ),
            text=build_text_config(
                LOCAL_STRUCTURE_SCHEMA,
                "douno_hierarchical_local_structure_g3",
            ),
            max_output_tokens=max_output_tokens,
            store=False,
        )
    except Exception as exc:
        return (
            LocalRegionStructure(
                region_id=scope.region.region_id,
                region_title=scope.region.semantic_title,
                region_confidence="LOW",
                sections=[],
                notes=["local model call failed"],
                interpretation_failed=True,
                diagnostics=local_scope_diagnostics(
                    scope,
                    failure_reason=f"Model structure request failed: {exc}",
                    elapsed_seconds=time.perf_counter() - started,
                ),
            ),
            ResponseDiagnostics(
                response_status=None,
                incomplete_reason=None,
                response_max_output_tokens=max_output_tokens,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                reasoning_tokens=None,
                raw_output_chars=0,
                output_item_statuses=[],
            ),
        )
    raw_text = response_text(response)
    diagnostics = extract_response_diagnostics(response, raw_text)
    try:
        parsed = parse_model_response(raw_text, diagnostics)
    except UniversalStructureError as exc:
        failure_reason = str(exc)
        return (
            LocalRegionStructure(
                region_id=scope.region.region_id,
                region_title=scope.region.semantic_title,
                region_confidence="LOW",
                sections=[],
                notes=["local model call failed"],
                interpretation_failed=True,
                diagnostics=local_scope_diagnostics(
                    scope,
                    response_diagnostics=diagnostics,
                    failure_reason=failure_reason,
                    elapsed_seconds=time.perf_counter() - started,
                    raw_output_chars=len(raw_text),
                ),
            ),
            diagnostics,
        )
    local = local_structure_from_json(scope.region.region_id, parsed)
    local = validate_local_candidate_sections(scope, local)
    return (
        LocalRegionStructure(
            **{
                field_name: getattr(local, field_name)
                for field_name in LocalRegionStructure.__dataclass_fields__
                if field_name != "diagnostics"
            },
            diagnostics=local_scope_diagnostics(
                scope,
                response_diagnostics=diagnostics,
                elapsed_seconds=time.perf_counter() - started,
                raw_output_chars=len(raw_text),
                structural_boundaries=len(local.sections),
            ),
        ),
        diagnostics,
    )


def reconnaissance_from_json(data: Dict[str, Any]) -> DocumentReconnaissance:
    regions = []
    for index, raw in enumerate(data.get("macro_regions") or [], start=1):
        regions.append(
            MacroRegion(
                region_id=f"region-{index:04d}",
                semantic_title=_required_text(raw.get("semantic_title"), f"Region {index}"),
                source_title=_optional_text(raw.get("source_title")),
                approximate_start=_position_from_json(raw.get("approximate_start")),
                approximate_end=_position_from_json(raw.get("approximate_end")),
                confidence=_enum(raw.get("confidence"), CONFIDENCE_VALUES, "LOW"),
                evidence_types=[str(item) for item in raw.get("evidence_types") or []],
                source_order=index,
            )
        )
    return DocumentReconnaissance(
        document_title=_required_text(data.get("document_title"), ""),
        document_type=_required_text(data.get("document_type"), ""),
        structure_confidence=_enum(data.get("structure_confidence"), CONFIDENCE_VALUES, "LOW"),
        organization_style=_required_text(data.get("organization_style"), "unknown"),
        macro_regions=regions,
        notes=list((data.get("notes") or [])[:5]),
    )


def local_structure_from_json(region_id: str, data: Dict[str, Any]) -> LocalRegionStructure:
    sections = []
    for index, raw in enumerate(data.get("sections") or [], start=1):
        sections.append(
            LocalSection(
                candidate_id=_optional_text(raw.get("candidate_id")),
                group_id=_optional_text(raw.get("group_id")),
                component_candidate_ids=[
                    str(item)
                    for item in (raw.get("component_candidate_ids") or [])
                    if str(item).strip()
                ],
                source_title=_optional_text(raw.get("source_title")),
                semantic_title=_required_text(
                    raw.get("semantic_title"),
                    _optional_text(raw.get("source_title")) or f"Section {index}",
                ),
                level=_coerce_int(raw.get("level"), 1, 1, 8),
                parent_title=_optional_text(raw.get("parent")),
                expected_start=_position_from_json(raw.get("expected_start")),
                expected_end=_position_from_json(raw.get("expected_end")),
                confidence=_enum(raw.get("confidence"), CONFIDENCE_VALUES, "LOW"),
                structural_role=_enum(raw.get("structural_role"), STRUCTURAL_ROLES, STRUCTURAL_ROLE_SECTION),
                classification=_local_classification_from_raw(raw),
                source_order=index,
            )
        )
    return LocalRegionStructure(
        region_id=region_id,
        region_title=_required_text(data.get("region_title"), region_id),
        region_confidence=_enum(data.get("region_confidence"), CONFIDENCE_VALUES, "LOW"),
        sections=sections,
        continuation_nodes=[
            LocalObservation(
                title=_required_text(raw.get("title"), "Continuation"),
                expected_start=_position_from_json(raw.get("expected_start")),
                confidence=_enum(raw.get("confidence"), CONFIDENCE_VALUES, "LOW"),
            )
            for raw in data.get("continuation_nodes") or []
        ],
        local_only_nodes=[
            LocalObservation(
                title=_required_text(raw.get("title"), "Local structure"),
                kind=_required_text(raw.get("kind"), "local"),
                expected_start=_position_from_json(raw.get("expected_start")),
                confidence=_enum(raw.get("confidence"), CONFIDENCE_VALUES, "LOW"),
            )
            for raw in data.get("local_only_nodes") or []
        ],
        notes=list((data.get("notes") or [])[:5]),
    )


def ground_macro_regions(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
) -> List[MacroRegionGrounding]:
    blocks = document.blocks
    occupied: set[Tuple[str, int, int, int, int]] = set()
    groundings: List[MacroRegionGrounding] = []
    previous_position: Optional[SourcePosition] = None
    for region in reconnaissance.macro_regions:
        grounding = _ground_single_region(blocks, region, occupied)
        if (
            grounding.source_position is not None
            and previous_position is not None
            and grounding.source_position < previous_position
        ):
            grounding = MacroRegionGrounding(
                region_id=region.region_id,
                source_title=region.source_title,
                semantic_title=region.semantic_title,
                source_position=None,
                block_list_index=None,
                method="source_order_rejected",
                confidence=0.0,
                fidelity="unresolved",
                status="unresolved",
                reason="grounding would violate macro-region source order",
            )
        if grounding.source_position:
            occupied.add(_position_key(grounding.source_position))
            previous_position = grounding.source_position
        groundings.append(grounding)
    return groundings


def prepare_local_analysis_scopes(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
) -> List[LocalAnalysisScope]:
    blocks = document.blocks
    if not blocks:
        return []

    grounded = [item for item in groundings if item.block_list_index is not None]
    if not grounded:
        if _is_small_document(blocks):
            region = MacroRegion(
                region_id="small-document-0001",
                semantic_title=reconnaissance.document_title or document.title,
                source_title=None,
                approximate_start=blocks[0].source_position,
                approximate_end=blocks[-1].source_position,
                confidence=reconnaissance.structure_confidence,
                evidence_types=["explicit_small_document_scope"],
                source_order=1,
                source_start_order=blocks[0].source_order,
                source_end_order=blocks[-1].source_order,
            )
            return [_make_scope("small-document-0001", "small_document", region, blocks)]
        return [
            _make_scope(region.region_id, "fallback_window", region, window_blocks)
            for region, window_blocks in fallback_windows(document)
        ]

    scopes: List[LocalAnalysisScope] = []
    sorted_grounded = sorted(grounded, key=lambda item: item.block_list_index or 0)
    grounded_by_id = {item.region_id: item for item in sorted_grounded}
    for index, grounding in enumerate(sorted_grounded):
        start = grounding.block_list_index or 0
        assert grounding.source_position is not None
        if index + 1 < len(sorted_grounded):
            next_grounding = sorted_grounded[index + 1]
            end = max(start, next_grounding.block_list_index or start)
            end_position = next_grounding.source_position
        else:
            end = len(blocks) - 1
            end_position = _block_end_position(blocks[-1])
        region = next(
            item for item in reconnaissance.macro_regions if item.region_id == grounding.region_id
        )
        bounded_region = MacroRegion(
            **{
                field_name: getattr(region, field_name)
                for field_name in MacroRegion.__dataclass_fields__
                if field_name not in {"source_start_order", "source_end_order"}
            },
            source_start_order=blocks[start].source_order,
            source_end_order=blocks[end].source_order,
        )
        owned_blocks = _slice_blocks_between(
            blocks,
            start_position=grounding.source_position,
            end_position=end_position,
        )
        scopes.extend(
            _subdivide_owned_blocks(
                bounded_region,
                owned_blocks,
                base_scope_id=grounding.region_id,
                total_blocks=len(blocks),
            )
        )

    unresolved = [item for item in groundings if item.region_id not in grounded_by_id]
    for grounding in unresolved:
        region = next(
            item for item in reconnaissance.macro_regions if item.region_id == grounding.region_id
        )
        scopes.append(
            LocalAnalysisScope(
                scope_id=f"{grounding.region_id}-unresolved",
                scope_type="unresolved_region",
                region=region,
                blocks=[],
                start_unit="n/a",
                end_unit="n/a",
                block_count=0,
                input_char_count=0,
                skipped=True,
                skip_reason=grounding.reason or "macro-region could not be physically grounded",
                macro_region_id=region.region_id,
                subdivision_reason="unresolved_macro_region",
            )
        )
    return scopes


def resolve_macro_region_blocks(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
) -> List[Tuple[MacroRegion, List[CanonicalBlock]]]:
    blocks = document.blocks
    if not blocks:
        return []
    if not reconnaissance.macro_regions:
        return fallback_windows(document)

    resolved: List[Tuple[MacroRegion, List[CanonicalBlock]]] = []
    ordered_regions = sorted(reconnaissance.macro_regions, key=lambda region: region.source_order)
    starts = []
    for region in ordered_regions:
        starts.append(_nearest_block_index(blocks, region.approximate_start))
    for index, region in enumerate(ordered_regions):
        start = starts[index] if starts[index] is not None else 0
        if index + 1 < len(starts) and starts[index + 1] is not None:
            end = max(start + 1, starts[index + 1])
        else:
            explicit_end = _nearest_block_index(blocks, region.approximate_end)
            end = explicit_end + 1 if explicit_end is not None else len(blocks)
        region_blocks = blocks[start:end] or blocks
        resolved.append(
            (
                MacroRegion(
                    **{
                        field_name: getattr(region, field_name)
                        for field_name in MacroRegion.__dataclass_fields__
                        if field_name not in {"source_start_order", "source_end_order"}
                    },
                    source_start_order=region_blocks[0].source_order,
                    source_end_order=region_blocks[-1].source_order,
                ),
                region_blocks,
            )
        )
    return resolved


def fallback_windows(document: CanonicalDocumentInput) -> List[Tuple[MacroRegion, List[CanonicalBlock]]]:
    blocks = document.blocks
    if not blocks:
        return []
    windows = []
    start = 0
    index = 1
    step = max(1, WINDOW_TARGET_BLOCKS - WINDOW_OVERLAP_BLOCKS)
    while start < len(blocks):
        end = min(len(blocks), start + WINDOW_TARGET_BLOCKS)
        window_blocks = blocks[start:end]
        region = MacroRegion(
            region_id=f"window-{index:04d}",
            semantic_title=f"Fallback window {index}",
            source_title=None,
            approximate_start=window_blocks[0].source_position,
            approximate_end=window_blocks[-1].source_position,
            confidence="LOW",
            evidence_types=["bounded_window_fallback"],
            source_order=index,
            source_start_order=window_blocks[0].source_order,
            source_end_order=window_blocks[-1].source_order,
        )
        windows.append((region, window_blocks))
        if end >= len(blocks):
            break
        start += step
        index += 1
    return windows


def discover_structural_candidates(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
) -> List[StructuralCandidate]:
    blocks = document.blocks
    macro_ranges = _macro_region_ranges(document, reconnaissance, groundings)
    candidates_by_key: Dict[Tuple[str, int, int, int, str], StructuralCandidate] = {}

    for block in blocks:
        text = (block.raw_text or block.text or "").strip()
        if not text:
            continue
        signals = _candidate_evidence_signals(block, text)
        if not signals:
            continue
        candidate = _candidate_from_block(
            block,
            text=text,
            signals=signals,
            macro_region_id=_candidate_macro_region_id(block.source_position, macro_ranges),
            source_order=len(candidates_by_key) + 1,
        )
        candidates_by_key[_candidate_key(candidate)] = candidate

    for grounding in groundings:
        if grounding.source_position is None or not grounding.source_title:
            continue
        block = _block_at_position(blocks, grounding.source_position)
        if block is None:
            continue
        literal = grounding.source_title.strip()
        if not literal:
            continue
        signals = ["global_macro_source_title", f"grounding:{grounding.method}"]
        if block.role_hint in {ROLE_HEADING, ROLE_TITLE}:
            signals.append(f"native_role:{block.role_hint}")
        candidate = _candidate_from_block(
            block,
            text=literal,
            signals=signals,
            macro_region_id=grounding.region_id,
            source_order=len(candidates_by_key) + 1,
            position=grounding.source_position,
            score_bonus=0.45,
        )
        candidates_by_key[_candidate_key(candidate)] = candidate

    ordered = sorted(candidates_by_key.values(), key=lambda item: _position_tuple(item.source_position))
    ordered_candidates = [
        StructuralCandidate(
            **{
                field_name: getattr(candidate, field_name)
                for field_name in StructuralCandidate.__dataclass_fields__
                if field_name not in {"candidate_id", "source_order", "physical_context"}
            },
            candidate_id=f"cand-{index:05d}",
            source_order=index,
        )
        for index, candidate in enumerate(ordered, start=1)
    ]
    return attach_candidate_physical_context(document, ordered_candidates)


def attach_candidate_physical_context(
    document: CanonicalDocumentInput,
    candidates: Sequence[StructuralCandidate],
) -> List[StructuralCandidate]:
    blocks = document.blocks
    block_index = {
        (
            block.source_position.unit_type,
            block.source_position.unit_index,
            block.source_position.block_index,
        ): index
        for index, block in enumerate(blocks)
    }
    enriched: List[StructuralCandidate] = []
    for index, candidate in enumerate(candidates):
        previous_candidate = candidates[index - 1] if index > 0 else None
        next_candidate = candidates[index + 1] if index + 1 < len(candidates) else None
        previous_distance = _candidate_block_distance(block_index, previous_candidate, candidate)
        next_distance = _candidate_block_distance(block_index, candidate, next_candidate)
        context = CandidatePhysicalContext(
            previous_candidate_id=previous_candidate.candidate_id if previous_candidate else None,
            next_candidate_id=next_candidate.candidate_id if next_candidate else None,
            previous_block_distance=previous_distance,
            next_block_distance=next_distance,
            same_unit_as_previous=_same_candidate_unit(previous_candidate, candidate) if previous_candidate else None,
            same_unit_as_next=_same_candidate_unit(candidate, next_candidate) if next_candidate else None,
            adjacent_to_previous=previous_distance == 1,
            adjacent_to_next=next_distance == 1,
            body_content_before_previous_candidate=_body_content_between(blocks, block_index, previous_candidate, candidate) if previous_candidate else None,
            body_content_before_next_candidate=_body_content_between(blocks, block_index, candidate, next_candidate) if next_candidate else None,
            compatible_with_previous=_candidates_native_compatible(previous_candidate, candidate) if previous_candidate else None,
            compatible_with_next=_candidates_native_compatible(candidate, next_candidate) if next_candidate else None,
        )
        enriched.append(replace(candidate, physical_context=context))
    return enriched


def propose_candidate_groups(
    document: CanonicalDocumentInput,
    candidates: Sequence[StructuralCandidate],
) -> List[CandidateGroupProposal]:
    proposals: List[CandidateGroupProposal] = []
    used_pairs: set[Tuple[str, ...]] = set()
    for first, second in zip(candidates, candidates[1:]):
        group = _compound_pair_proposal(first, second, len(proposals) + 1)
        if group and tuple(group.candidate_ids) not in used_pairs:
            proposals.append(group)
            used_pairs.add(tuple(group.candidate_ids))

    for index, candidate in enumerate(candidates, start=len(proposals) + 1):
        proposals.append(
            CandidateGroupProposal(
                group_id=f"group-{index:05d}",
                candidate_ids=[candidate.candidate_id],
                source_start=candidate.source_position,
                source_end=replace(
                    candidate.source_position,
                    start_offset=candidate.end_offset or candidate.source_position.end_offset or candidate.source_position.start_offset,
                    end_offset=candidate.end_offset or candidate.source_position.end_offset,
                ),
                relationship_hypothesis="single_structural_component",
                confidence="HIGH" if candidate.candidate_score >= 0.65 else "MEDIUM",
                evidence_signals=["single_candidate_proposal"],
                primary_candidate_id=candidate.candidate_id,
                macro_region_id=candidate.macro_region_id,
            )
        )
    return proposals


def _compound_pair_proposal(
    first: StructuralCandidate,
    second: StructuralCandidate,
    index: int,
) -> Optional[CandidateGroupProposal]:
    if not _adjacent_without_body(first, second):
        return None
    if first.macro_region_id != second.macro_region_id:
        return None
    if not _candidates_native_compatible(first, second):
        return None
    first_marker = _marker_like_candidate(first)
    second_marker = _marker_like_candidate(second)
    first_descriptive = _descriptive_candidate(second)
    if first_marker and first_descriptive:
        hypothesis = "compound_heading_components"
        confidence = "HIGH" if _strong_candidate_native_evidence(first) and _strong_candidate_native_evidence(second) else "MEDIUM"
        primary = second.candidate_id
    elif _possible_title_subtitle_pair(first, second):
        hypothesis = "ambiguous_grouping"
        confidence = "MEDIUM"
        primary = first.candidate_id
    else:
        return None
    return CandidateGroupProposal(
        group_id=f"group-{index:05d}",
        candidate_ids=[first.candidate_id, second.candidate_id],
        source_start=first.source_position,
        source_end=replace(
            second.source_position,
            start_offset=second.end_offset or second.source_position.end_offset or second.source_position.start_offset,
            end_offset=second.end_offset or second.source_position.end_offset,
        ),
        relationship_hypothesis=hypothesis,
        confidence=confidence,
        evidence_signals=[
            "adjacent_candidates",
            "no_intervening_body",
            "compatible_native_hierarchy",
            "marker_plus_descriptive" if first_marker else "title_subtitle_like",
        ],
        primary_candidate_id=primary,
        macro_region_id=first.macro_region_id,
    )


def prepare_candidate_analysis_scopes(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
    candidates: Sequence[StructuralCandidate],
    candidate_groups: Sequence[CandidateGroupProposal] = (),
    *,
    max_model_calls: int,
) -> List[LocalAnalysisScope]:
    if not candidates:
        return _fallback_candidate_scopes(document, reconnaissance, max_model_calls=max_model_calls)

    regions_by_id = {region.region_id: region for region in reconnaissance.macro_regions}
    grouped: Dict[str, List[StructuralCandidate]] = {}
    for candidate in candidates:
        if candidate.macro_region_id and candidate.macro_region_id in regions_by_id:
            grouped.setdefault(candidate.macro_region_id, []).append(candidate)
    groups_by_region: Dict[str, List[CandidateGroupProposal]] = {}
    for group in candidate_groups:
        if group.macro_region_id and group.macro_region_id in regions_by_id and not group.rejected:
            groups_by_region.setdefault(group.macro_region_id, []).append(group)

    scopes: List[LocalAnalysisScope] = []
    for region in reconnaissance.macro_regions:
        region_candidates = grouped.get(region.region_id, [])
        if not region_candidates:
            grounding = next((item for item in groundings if item.region_id == region.region_id), None)
            scopes.append(
                LocalAnalysisScope(
                    scope_id=f"{region.region_id}-no-candidates",
                    scope_type="candidate_region_empty",
                    region=region,
                    blocks=[],
                    start_unit="n/a",
                    end_unit="n/a",
                    block_count=0,
                    input_char_count=0,
                    skipped=True,
                    skip_reason=(
                        "no deterministic structural candidates found in grounded macro-region"
                        if grounding and grounding.status == "grounded"
                        else "macro-region could not be physically grounded"
                    ),
                    macro_region_id=region.region_id,
                    subdivision_reason="candidate_discovery_empty",
                )
            )
            continue
        for batch_index, batch in enumerate(_batch_candidates(region_candidates), start=1):
            exhausted = len([scope for scope in scopes if not scope.skipped]) >= max_model_calls
            batch_candidate_ids = {candidate.candidate_id for candidate in batch}
            scopes.append(_candidate_scope(
                region,
                batch,
                [
                    group for group in groups_by_region.get(region.region_id, [])
                    if all(candidate_id in batch_candidate_ids for candidate_id in group.candidate_ids)
                ],
                batch_index,
                exhausted=exhausted,
            ))
    return scopes


def reconcile_macro_boundaries(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
    candidates: Sequence[StructuralCandidate],
    candidate_groups: Sequence[CandidateGroupProposal] = (),
) -> MacroBoundaryReconciliation:
    original_regions = list(reconnaissance.macro_regions)
    original_groundings = list(groundings)
    grounded_by_id = {item.region_id: item for item in original_groundings if item.source_position}
    existing_positions = {
        _position_boundary_key(item.source_position)
        for item in original_groundings
        if item.source_position is not None
    }
    macro_level_by_region = _grounded_macro_levels(document, reconnaissance, original_groundings)
    compound_component_ids = {
        candidate_id
        for group in candidate_groups
        if group.relationship_hypothesis == "compound_heading_components" and not group.rejected
        for candidate_id in group.candidate_ids
    }
    proposals = []
    accepted: List[Tuple[StructuralCandidate, str, Optional[CandidateGroupProposal]]] = []
    duplicate_suppressed = 0

    for candidate in candidates:
        group = _macro_group_for_candidate(candidate, candidate_groups)
        if candidate.candidate_id in compound_component_ids and group and candidate.candidate_id != group.primary_candidate_id:
            reason = "component_of_compound_heading_group"
        else:
            reason = _macro_boundary_rejection_reason(
                candidate,
                grounded_by_id,
                existing_positions,
                macro_level_by_region,
            )
        proposal = {
            "candidate_id": candidate.candidate_id,
            "group_id": group.group_id if group else None,
            "source_title": candidate.source_title,
            "source_position": _position_label(candidate.source_position),
            "macro_region_id": candidate.macro_region_id,
            "evidence_signals": list(candidate.evidence_signals),
            "native_hierarchy_hint": candidate.native_hierarchy_hint,
            "candidate_score": candidate.candidate_score,
            "accepted": reason is None,
            "rejection_reason": reason,
        }
        proposals.append(proposal)
        if reason == "duplicate_existing_macro_boundary":
            duplicate_suppressed += 1
        if reason is None:
            accepted.append((candidate, candidate.macro_region_id or "", group))

    if not accepted:
        diagnostics = _macro_reconciliation_diagnostics(
            original_regions,
            original_groundings,
            original_regions,
            original_groundings,
            proposals,
            duplicate_suppressed,
        )
        return MacroBoundaryReconciliation(
            reconnaissance=reconnaissance,
            groundings=list(groundings),
            candidates=list(candidates),
            candidate_groups=list(candidate_groups),
            diagnostics=diagnostics,
        )

    new_regions = list(original_regions)
    next_region_index = len(new_regions) + 1
    new_groundings = list(original_groundings)
    for candidate, absorbed_region_id, group in sorted(accepted, key=lambda item: _position_tuple(item[0].source_position)):
        source_title = candidate.source_title
        semantic_title = candidate.source_title
        evidence_types = list(candidate.evidence_signals)
        if group and group.relationship_hypothesis == "compound_heading_components":
            evidence_types.extend(["logical_node_from_compound_heading", *group.evidence_signals])
        region_id = f"reconciled-region-{next_region_index:04d}"
        next_region_index += 1
        new_regions.append(
            MacroRegion(
                region_id=region_id,
                semantic_title=semantic_title,
                source_title=source_title,
                approximate_start=candidate.source_position,
                approximate_end=None,
                confidence="HIGH" if _strong_candidate_native_evidence(candidate) else "MEDIUM",
                evidence_types=[
                    "deterministic_macro_boundary_reconciliation",
                    *evidence_types,
                    f"split_from:{absorbed_region_id}",
                ],
                source_order=len(new_regions) + 1,
                source_start_order=candidate.source_order,
                source_end_order=None,
            )
        )
        new_groundings.append(
            MacroRegionGrounding(
                region_id=region_id,
                source_title=candidate.source_title,
                semantic_title=candidate.source_title,
                source_position=candidate.source_position,
                block_list_index=_block_index_for_position(document.blocks, candidate.source_position),
                method="deterministic_candidate_macro_boundary",
                confidence=candidate.candidate_score,
                fidelity="exact",
                status="grounded",
                reason=f"candidate split/rebounded macro-region {absorbed_region_id}",
            )
        )

    ordered = sorted(
        zip(new_regions, new_groundings),
        key=lambda pair: (
            1 if pair[1].source_position is None else 0,
            _position_tuple(pair[1].source_position) if pair[1].source_position else ("zz", 10**9, 10**9, 10**9, 10**9),
            pair[0].source_order,
        ),
    )
    rewritten_regions: List[MacroRegion] = []
    rewritten_groundings: List[MacroRegionGrounding] = []
    for index, (region, grounding) in enumerate(ordered, start=1):
        rewritten_regions.append(
            MacroRegion(
                **{
                    field_name: getattr(region, field_name)
                    for field_name in MacroRegion.__dataclass_fields__
                    if field_name != "source_order"
                },
                source_order=index,
            )
        )
        rewritten_groundings.append(grounding)

    reconciled_reconnaissance = DocumentReconnaissance(
        document_title=reconnaissance.document_title,
        document_type=reconnaissance.document_type,
        structure_confidence=reconnaissance.structure_confidence,
        organization_style=reconnaissance.organization_style,
        macro_regions=rewritten_regions,
        notes=list(reconnaissance.notes),
    )
    reassigned_candidates = _assign_candidates_to_reconciled_regions(
        document,
        reconciled_reconnaissance,
        rewritten_groundings,
        candidates,
    )
    reassigned_groups = _assign_groups_to_reconciled_regions(
        document,
        reconciled_reconnaissance,
        rewritten_groundings,
        candidate_groups,
    )
    diagnostics = _macro_reconciliation_diagnostics(
        original_regions,
        original_groundings,
        rewritten_regions,
        rewritten_groundings,
        proposals,
        duplicate_suppressed,
    )
    return MacroBoundaryReconciliation(
        reconnaissance=reconciled_reconnaissance,
        groundings=rewritten_groundings,
        candidates=reassigned_candidates,
        candidate_groups=reassigned_groups,
        diagnostics=diagnostics,
    )


def _macro_boundary_rejection_reason(
    candidate: StructuralCandidate,
    grounded_by_id: Dict[str, MacroRegionGrounding],
    existing_positions: set[Tuple[str, int, int, int]],
    macro_level_by_region: Dict[str, Optional[int]],
) -> Optional[str]:
    if candidate.macro_region_id is None:
        return "candidate_outside_grounded_macro_region"
    if candidate.macro_region_id not in grounded_by_id:
        return "owning_macro_region_not_grounded"
    if _position_boundary_key(candidate.source_position) in existing_positions:
        return "duplicate_existing_macro_boundary"
    if not _candidate_has_strong_macro_boundary_evidence(candidate):
        return "insufficient_native_or_source_evidence"
    current_level = macro_level_by_region.get(candidate.macro_region_id)
    if current_level is not None and candidate.native_hierarchy_hint is not None:
        if candidate.native_hierarchy_hint > current_level:
            return "nested_below_current_macro_level"
    elif not _strong_candidate_native_evidence(candidate):
        return "missing_compatible_hierarchy_evidence"
    return None


def _candidate_has_strong_macro_boundary_evidence(candidate: StructuralCandidate) -> bool:
    if "global_macro_source_title" in candidate.evidence_signals:
        return True
    return _strong_candidate_native_evidence(candidate)


def _strong_candidate_native_evidence(candidate: StructuralCandidate) -> bool:
    signals = set(candidate.evidence_signals)
    if candidate.role_hint in {ROLE_TITLE, ROLE_HEADING} and candidate.native_hierarchy_hint is not None:
        return True
    if "native_style_heading" in signals and candidate.native_hierarchy_hint is not None:
        return True
    if "native_hierarchy_hint" in signals and candidate.candidate_score >= 0.55:
        return True
    return False


def _grounded_macro_levels(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
) -> Dict[str, Optional[int]]:
    result: Dict[str, Optional[int]] = {}
    blocks = document.blocks
    for grounding in groundings:
        if grounding.source_position is None:
            result[grounding.region_id] = None
            continue
        block = _block_at_position(blocks, grounding.source_position)
        result[grounding.region_id] = block.native_hierarchy_hint if block else None
    return result


def _assign_candidates_to_reconciled_regions(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
    candidates: Sequence[StructuralCandidate],
) -> List[StructuralCandidate]:
    ranges = _macro_region_ranges(document, reconnaissance, groundings)
    return [
        replace(
            candidate,
            macro_region_id=_candidate_macro_region_id(candidate.source_position, ranges),
        )
        for candidate in candidates
    ]


def _assign_groups_to_reconciled_regions(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
    candidate_groups: Sequence[CandidateGroupProposal],
) -> List[CandidateGroupProposal]:
    ranges = _macro_region_ranges(document, reconnaissance, groundings)
    return [
        replace(
            group,
            macro_region_id=_candidate_macro_region_id(group.source_start, ranges),
        )
        for group in candidate_groups
    ]


def _macro_group_for_candidate(
    candidate: StructuralCandidate,
    groups: Sequence[CandidateGroupProposal],
) -> Optional[CandidateGroupProposal]:
    matches = [
        group for group in groups
        if not group.rejected
        and group.relationship_hypothesis == "compound_heading_components"
        and candidate.candidate_id in group.candidate_ids
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda group: (len(group.candidate_ids), group.group_id), reverse=True)[0]


def _macro_reconciliation_diagnostics(
    before_regions: Sequence[MacroRegion],
    before_groundings: Sequence[MacroRegionGrounding],
    after_regions: Sequence[MacroRegion],
    after_groundings: Sequence[MacroRegionGrounding],
    proposals: Sequence[Dict[str, Any]],
    duplicate_suppressed: int,
) -> Dict[str, Any]:
    accepted = [proposal for proposal in proposals if proposal.get("accepted")]
    rejected = [proposal for proposal in proposals if not proposal.get("accepted")]
    return {
        "macro_regions_before": [
            {
                "region_id": region.region_id,
                "semantic_title": region.semantic_title,
                "source_title": region.source_title,
                "source_order": region.source_order,
                "grounding": _grounding_summary(before_groundings, region.region_id),
            }
            for region in before_regions
        ],
        "strong_candidate_macro_boundary_proposals": list(proposals),
        "accepted_candidate_macro_boundaries": accepted,
        "rejected_candidate_macro_boundaries": rejected,
        "macro_regions_after": [
            {
                "region_id": region.region_id,
                "semantic_title": region.semantic_title,
                "source_title": region.source_title,
                "source_order": region.source_order,
                "grounding": _grounding_summary(after_groundings, region.region_id),
            }
            for region in after_regions
        ],
        "physical_scope_changes": len(after_regions) - len(before_regions),
        "duplicate_proposals_suppressed": duplicate_suppressed,
        "unresolved_semantic_macro_regions_preserved": sum(
            1 for item in after_groundings if item.source_position is None
        ),
    }


def candidate_grouping_diagnostics(groups: Sequence[CandidateGroupProposal]) -> Dict[str, Any]:
    multi = [group for group in groups if len(group.candidate_ids) > 1]
    single = [group for group in groups if len(group.candidate_ids) == 1]
    ambiguous = [group for group in groups if group.relationship_hypothesis == "ambiguous_grouping"]
    rejected = [group for group in groups if group.rejected]
    return {
        "group_proposals": len(groups),
        "single_candidate_proposals": len(single),
        "multi_candidate_proposals": len(multi),
        "ambiguous_proposals": len(ambiguous),
        "rejected_proposals": len(rejected),
        "multi_component_groups": [
            {
                "group_id": group.group_id,
                "candidate_ids": group.candidate_ids,
                "primary_candidate_id": group.primary_candidate_id,
                "relationship_hypothesis": group.relationship_hypothesis,
                "confidence": group.confidence,
                "evidence_signals": group.evidence_signals,
            }
            for group in multi[:25]
        ],
    }


def _grounding_summary(
    groundings: Sequence[MacroRegionGrounding],
    region_id: str,
) -> Dict[str, Any]:
    grounding = next((item for item in groundings if item.region_id == region_id), None)
    if grounding is None:
        return {"status": "missing"}
    return {
        "status": grounding.status,
        "method": grounding.method,
        "position": _position_label(grounding.source_position),
        "reason": grounding.reason,
    }


def _fallback_candidate_scopes(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    *,
    max_model_calls: int,
) -> List[LocalAnalysisScope]:
    if _is_small_document(document.blocks):
        windows = fallback_windows(document)[:max_model_calls]
    else:
        windows = fallback_windows(document)[: min(max_model_calls, 3)]
    scopes = []
    for region, blocks in windows:
        scopes.append(
            _apply_local_scope_guard(
                _make_scope(
                    region.region_id,
                    "fallback_window",
                    region,
                    blocks,
                    subdivision_reason="candidate_discovery_fallback_window",
                ),
                total_blocks=len(document.blocks),
            )
        )
    return scopes


def _candidate_scope(
    region: MacroRegion,
    candidates: Sequence[StructuralCandidate],
    candidate_groups: Sequence[CandidateGroupProposal],
    batch_index: int,
    *,
    exhausted: bool,
) -> LocalAnalysisScope:
    source_start = candidates[0].source_position if candidates else None
    source_end = candidates[-1].source_position if candidates else None
    context_chars = len(build_candidate_context_text(candidates)) + len(build_candidate_group_context_text(candidate_groups))
    return LocalAnalysisScope(
        scope_id=f"{region.region_id}-candidate-batch-{batch_index:03d}",
        scope_type="candidate_batch",
        region=region,
        blocks=[],
        start_unit=_position_unit_label(source_start),
        end_unit=_position_unit_label(source_end),
        block_count=0,
        input_char_count=context_chars,
        skipped=exhausted,
        skip_reason="hierarchical model-call budget exhausted" if exhausted else None,
        macro_region_id=region.region_id,
        subscope_index=batch_index,
        subdivision_reason="candidate_batch",
        source_start=source_start,
        source_end=source_end,
        candidates=list(candidates),
        candidate_groups=list(candidate_groups),
        call_budget_exhausted=exhausted,
    )


def _batch_candidates(candidates: Sequence[StructuralCandidate]) -> List[List[StructuralCandidate]]:
    batches: List[List[StructuralCandidate]] = []
    current: List[StructuralCandidate] = []
    for candidate in candidates:
        trial = current + [candidate]
        if (
            current
            and (
                len(trial) > MAX_CANDIDATES_PER_LOCAL_CALL
                or len(build_candidate_context_text(trial)) > MAX_CANDIDATE_CONTEXT_CHARS
            )
        ):
            batches.append(current)
            current = [candidate]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def _candidate_evidence_signals(block: CanonicalBlock, text: str) -> List[str]:
    signals: List[str] = []
    style = (block.style_hint or "").lower()
    stripped = text.strip()
    if block.role_hint in {ROLE_HEADING, ROLE_TITLE}:
        signals.append(f"native_role:{block.role_hint}")
    if "heading" in style or "title" in style:
        signals.append("native_style_heading")
    if block.native_hierarchy_hint is not None and block.role_hint != ROLE_LIST:
        signals.append("native_hierarchy_hint")
    numbering = _numbering_pattern(stripped)
    if numbering:
        signals.append(f"numbering:{numbering}")
    if _short_heading_like_text(block, stripped):
        signals.append("short_heading_like_text")
    if _unit_beginning_candidate(block, stripped):
        signals.append("unit_beginning")
    return signals


def _candidate_block_distance(
    block_index: Dict[Tuple[str, int, int], int],
    left: Optional[StructuralCandidate],
    right: Optional[StructuralCandidate],
) -> Optional[int]:
    if left is None or right is None:
        return None
    left_index = block_index.get((left.unit_type, left.unit_index, left.block_index))
    right_index = block_index.get((right.unit_type, right.unit_index, right.block_index))
    if left_index is None or right_index is None:
        return None
    return right_index - left_index


def _same_candidate_unit(
    left: Optional[StructuralCandidate],
    right: Optional[StructuralCandidate],
) -> bool:
    return bool(
        left
        and right
        and left.unit_type == right.unit_type
        and left.unit_index == right.unit_index
    )


def _body_content_between(
    blocks: Sequence[CanonicalBlock],
    block_index: Dict[Tuple[str, int, int], int],
    left: Optional[StructuralCandidate],
    right: Optional[StructuralCandidate],
) -> Optional[bool]:
    if left is None or right is None:
        return None
    left_index = block_index.get((left.unit_type, left.unit_index, left.block_index))
    right_index = block_index.get((right.unit_type, right.unit_index, right.block_index))
    if left_index is None or right_index is None or right_index <= left_index:
        return None
    between = blocks[left_index + 1:right_index]
    return any(block.role_hint in {ROLE_BODY, ROLE_LIST, ROLE_TABLE} and (block.raw_text or block.text or "").strip() for block in between)


def _candidates_native_compatible(
    left: Optional[StructuralCandidate],
    right: Optional[StructuralCandidate],
) -> bool:
    if left is None or right is None:
        return False
    if left.native_hierarchy_hint is not None and right.native_hierarchy_hint is not None:
        if abs(left.native_hierarchy_hint - right.native_hierarchy_hint) > 0:
            return False
    if left.role_hint not in {ROLE_HEADING, ROLE_TITLE} or right.role_hint not in {ROLE_HEADING, ROLE_TITLE}:
        return False
    return True


def _adjacent_without_body(first: StructuralCandidate, second: StructuralCandidate) -> bool:
    context = first.physical_context
    if not context:
        return False
    return (
        context.next_candidate_id == second.candidate_id
        and context.adjacent_to_next
        and context.body_content_before_next_candidate is False
    )


def _marker_like_candidate(candidate: StructuralCandidate) -> bool:
    text = candidate.source_title.strip()
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) <= 2 and (candidate.numbering_pattern or re.search(r"\d", text)):
        return True
    if len(text) <= 18 and (
        bool(re.search(r"\d", text))
        or bool(re.fullmatch(r"[IVXLCDM]+", text, flags=re.IGNORECASE))
    ):
        return True
    return False


def _descriptive_candidate(candidate: StructuralCandidate) -> bool:
    words = re.findall(r"\w+", candidate.source_title, flags=re.UNICODE)
    if _marker_like_candidate(candidate):
        return False
    return len(words) >= 3 or (len(words) == 1 and len(words[0]) >= 5)


def _possible_title_subtitle_pair(first: StructuralCandidate, second: StructuralCandidate) -> bool:
    return (
        first.unit_type == "slide"
        and first.unit_index == second.unit_index
        and first.role_hint == ROLE_TITLE
        and second.role_hint in {ROLE_HEADING, ROLE_TITLE}
        and _adjacent_without_body(first, second)
    )


def _candidate_from_block(
    block: CanonicalBlock,
    *,
    text: str,
    signals: Sequence[str],
    macro_region_id: Optional[str],
    source_order: int,
    position: Optional[SourcePosition] = None,
    score_bonus: float = 0.0,
) -> StructuralCandidate:
    source_position = position or block.source_position
    end_offset = source_position.end_offset
    if end_offset is None and source_position.start_offset is not None:
        end_offset = source_position.start_offset + len(text)
    score = min(1.0, score_bonus + sum(_signal_weight(signal) for signal in signals))
    return StructuralCandidate(
        candidate_id=f"candidate-temp-{source_order:05d}",
        source_title=text.strip(),
        source_position=source_position,
        source_order=source_order,
        unit_type=source_position.unit_type,
        unit_index=source_position.unit_index,
        block_index=source_position.block_index,
        start_offset=source_position.start_offset,
        end_offset=end_offset,
        role_hint=block.role_hint,
        style_hint=block.style_hint,
        native_hierarchy_hint=block.native_hierarchy_hint,
        text_length=len(text),
        numbering_pattern=_numbering_pattern(text),
        evidence_signals=list(dict.fromkeys(signals)),
        candidate_score=round(score, 3),
        context_before=_neighbor_text(block, -1),
        context_after=_neighbor_text(block, 1),
        macro_region_id=macro_region_id,
    )


def _signal_weight(signal: str) -> float:
    if signal.startswith("global_macro_source_title"):
        return 0.45
    if signal.startswith("native_role:title"):
        return 0.42
    if signal.startswith("native_role:heading"):
        return 0.38
    if signal.startswith("native_style_heading"):
        return 0.32
    if signal.startswith("native_hierarchy_hint"):
        return 0.28
    if signal.startswith("numbering:"):
        return 0.28
    if signal == "short_heading_like_text":
        return 0.18
    if signal == "unit_beginning":
        return 0.08
    if signal.startswith("grounding:"):
        return 0.12
    return 0.05


def _short_heading_like_text(block: CanonicalBlock, text: str) -> bool:
    if block.role_hint in {ROLE_LIST, ROLE_TABLE}:
        return False
    if len(text) > 140 or len(text) < 3:
        return False
    if "\n" in text and len([line for line in text.splitlines() if line.strip()]) > 2:
        return False
    if text.endswith((".", ",", ";")) and not _numbering_pattern(text):
        return False
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) > 14:
        return False
    alpha = [char for char in text if char.isalpha()]
    if alpha and sum(1 for char in alpha if char.isupper()) / max(1, len(alpha)) > 0.55:
        return True
    if words and sum(1 for word in words if word[:1].isupper()) >= max(1, len(words) // 2):
        return True
    return bool(_numbering_pattern(text))


def _unit_beginning_candidate(block: CanonicalBlock, text: str) -> bool:
    return block.source_position.block_index == 1 and len(text) <= 120 and block.role_hint != ROLE_TABLE


def _numbering_pattern(text: str) -> Optional[str]:
    stripped = text.strip()
    if re.match(r"^\d+(?:\.\d+){1,5}\s+\S", stripped):
        return "dotted"
    if re.match(r"^\d{1,2}[.)]?\s+[A-ZÀ-ÖØ-Þ0-9][^\n]{2,140}$", stripped):
        return "integer"
    if re.match(r"^[A-Z][.)]\s+\S", stripped):
        return "alpha"
    return None


def _neighbor_text(block: CanonicalBlock, direction: int) -> Optional[str]:
    return None


def _candidate_key(candidate: StructuralCandidate) -> Tuple[str, int, int, int, str]:
    return (
        candidate.unit_type,
        candidate.unit_index,
        candidate.block_index,
        candidate.start_offset,
        _title_key(candidate.source_title),
    )


def _block_at_position(
    blocks: Sequence[CanonicalBlock],
    position: SourcePosition,
) -> Optional[CanonicalBlock]:
    for block in blocks:
        if (
            block.source_position.unit_type == position.unit_type
            and block.source_position.unit_index == position.unit_index
            and block.source_position.block_index == position.block_index
        ):
            return block
    return None


def _macro_region_ranges(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    groundings: Sequence[MacroRegionGrounding],
) -> List[Tuple[str, SourcePosition, SourcePosition]]:
    blocks = document.blocks
    if not blocks:
        return []
    grounded = sorted(
        [item for item in groundings if item.source_position is not None],
        key=lambda item: _position_tuple(item.source_position),
    )
    ranges = []
    for index, grounding in enumerate(grounded):
        assert grounding.source_position is not None
        if index + 1 < len(grounded):
            assert grounded[index + 1].source_position is not None
            end = grounded[index + 1].source_position
        else:
            end = _block_end_position(blocks[-1])
        ranges.append((grounding.region_id, grounding.source_position, end))
    return ranges


def _candidate_macro_region_id(
    position: SourcePosition,
    macro_ranges: Sequence[Tuple[str, SourcePosition, SourcePosition]],
) -> Optional[str]:
    for region_id, start, end in macro_ranges:
        if _position_lte(start, position) and _position_lt(position, end):
            return region_id
    return None


def _position_lte(left: SourcePosition, right: SourcePosition) -> bool:
    return _position_tuple(left) <= _position_tuple(right)


def _position_lt(left: SourcePosition, right: SourcePosition) -> bool:
    return _position_tuple(left) < _position_tuple(right)


def _position_tuple(position: SourcePosition) -> Tuple[str, int, int, int, int]:
    start = position.start_offset if position.start_offset is not None else 0
    end = position.end_offset if position.end_offset is not None else start
    return (
        position.unit_type,
        position.unit_index,
        position.block_index,
        start,
        end,
    )


def _position_boundary_key(position: SourcePosition) -> Tuple[str, int, int, int]:
    return (
        position.unit_type,
        position.unit_index,
        position.block_index,
        position.start_offset if position.start_offset is not None else 0,
    )


def _block_index_for_position(
    blocks: Sequence[CanonicalBlock],
    position: SourcePosition,
) -> Optional[int]:
    for index, block in enumerate(blocks):
        if (
            block.source_position.unit_type == position.unit_type
            and block.source_position.unit_index == position.unit_index
            and block.source_position.block_index == position.block_index
        ):
            return index
    return None


def build_candidate_context(scope: LocalAnalysisScope) -> str:
    return (
        "Analyze only the candidate list below. Each candidate has literal source "
        "text and physical provenance. Accept candidates only if they are genuine "
        "document-structure boundaries. Candidate group proposals may represent "
        "one logical heading made from multiple physical components, a possible "
        "hierarchical stack, or an ambiguous relationship. Use group_id when one "
        "group proposal represents a single logical structural node; otherwise use "
        "candidate_id exactly as supplied.\n\n"
        + build_candidate_context_text(scope.candidates)
        + "\n\n"
        + build_candidate_group_context_text(scope.candidate_groups)
    )


def build_candidate_context_text(candidates: Sequence[StructuralCandidate]) -> str:
    lines = ["STRUCTURAL_CANDIDATES"]
    for candidate in candidates:
        evidence = ",".join(candidate.evidence_signals)
        before = _compact_context(candidate.context_before)
        after = _compact_context(candidate.context_after)
        physical = candidate.physical_context
        physical_summary = (
            f"prev={physical.previous_candidate_id or 'n/a'}:{physical.previous_block_distance} "
            f"next={physical.next_candidate_id or 'n/a'}:{physical.next_block_distance} "
            f"adj_prev={physical.adjacent_to_previous} adj_next={physical.adjacent_to_next} "
            f"body_prev={physical.body_content_before_previous_candidate} "
            f"body_next={physical.body_content_before_next_candidate} "
            f"native_prev={physical.compatible_with_previous} "
            f"native_next={physical.compatible_with_next}"
            if physical
            else "n/a"
        )
        lines.append(
            f"- id={candidate.candidate_id} | text={json.dumps(candidate.source_title, ensure_ascii=False)} | "
            f"pos={candidate.unit_type}:{candidate.unit_index}:{candidate.block_index}@{candidate.start_offset}:{candidate.end_offset} | "
            f"role={candidate.role_hint} | style={candidate.style_hint or 'n/a'} | "
            f"hierarchy_hint={candidate.native_hierarchy_hint if candidate.native_hierarchy_hint is not None else 'n/a'} | "
            f"numbering={candidate.numbering_pattern or 'n/a'} | score={candidate.candidate_score} | "
            f"evidence={evidence or 'n/a'} | physical={physical_summary} | before={before} | after={after}"
        )
    return "\n".join(lines)


def build_candidate_group_context_text(groups: Sequence[CandidateGroupProposal]) -> str:
    lines = ["CANDIDATE_GROUP_PROPOSALS"]
    for group in groups:
        lines.append(
            f"- id={group.group_id} | candidates={','.join(group.candidate_ids)} | "
            f"hypothesis={group.relationship_hypothesis} | confidence={group.confidence} | "
            f"primary={group.primary_candidate_id} | evidence={','.join(group.evidence_signals)} | "
            f"span={_position_label(group.source_start)}->{_position_label(group.source_end)}"
        )
    return "\n".join(lines)


def _compact_context(value: Optional[str]) -> str:
    if not value:
        return "n/a"
    text = " ".join(value.split())
    return json.dumps(text[:120], ensure_ascii=False)


def validate_local_candidate_sections(
    scope: LocalAnalysisScope,
    local: LocalRegionStructure,
) -> LocalRegionStructure:
    if not scope.candidates:
        return local
    by_id = {candidate.candidate_id: candidate for candidate in scope.candidates}
    by_title = {_title_key(candidate.source_title): candidate for candidate in scope.candidates}
    groups_by_id = {
        group.group_id: group
        for group in scope.candidate_groups
        if not group.rejected
    }
    validated: List[LocalSection] = []
    for section in local.sections:
        group = groups_by_id.get(section.group_id or "")
        candidate = by_id.get(group.primary_candidate_id) if group else None
        component_candidate_ids: List[str] = list(group.candidate_ids) if group else []
        expected_start = candidate.source_position if candidate else None
        expected_end = group.source_end if group else None
        if candidate is None:
            candidate = by_id.get(section.candidate_id or "")
            if candidate is not None:
                component_candidate_ids = [candidate.candidate_id]
                expected_start = candidate.source_position
                expected_end = replace(
                    candidate.source_position,
                    start_offset=candidate.end_offset or candidate.source_position.end_offset or candidate.source_position.start_offset,
                    end_offset=candidate.end_offset or candidate.source_position.end_offset,
                )
        if candidate is None and section.source_title:
            candidate = by_title.get(_title_key(section.source_title))
            if candidate is not None:
                component_candidate_ids = [candidate.candidate_id]
                expected_start = candidate.source_position
                expected_end = replace(
                    candidate.source_position,
                    start_offset=candidate.end_offset or candidate.source_position.end_offset or candidate.source_position.start_offset,
                    end_offset=candidate.end_offset or candidate.source_position.end_offset,
                )
        if candidate is None:
            validated.append(
                replace(
                    section,
                    candidate_id=None,
                    group_id=None,
                    component_candidate_ids=[],
                    source_title=None,
                    expected_start=None,
                    expected_end=None,
                    classification=LOCAL_CLASS_NON_STRUCTURAL,
                    structural_role=STRUCTURAL_ROLE_LOCAL,
                )
            )
            continue
        validated.append(
            replace(
                section,
                candidate_id=candidate.candidate_id,
                group_id=group.group_id if group else section.group_id,
                component_candidate_ids=component_candidate_ids or [candidate.candidate_id],
                source_title=candidate.source_title,
                semantic_title=candidate.source_title,
                expected_start=expected_start or candidate.source_position,
                expected_end=expected_end,
            )
        )
    return replace(local, sections=validated)


def local_scope_diagnostics(
    scope: LocalAnalysisScope,
    *,
    response_diagnostics: Optional[ResponseDiagnostics] = None,
    failure_reason: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    raw_output_chars: Optional[int] = None,
    structural_boundaries: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "scope_id": scope.scope_id,
        "scope_type": scope.scope_type,
        "macro_region_id": scope.macro_region_id or scope.region.region_id,
        "subscope_index": scope.subscope_index,
        "subdivision_reason": scope.subdivision_reason,
        "region_id": scope.region.region_id,
        "source_unit_start": scope.start_unit,
        "source_unit_end": scope.end_unit,
        "source_start": _position_label(scope.source_start),
        "source_end": _position_label(scope.source_end),
        "block_count": scope.block_count,
        "input_char_count": scope.input_char_count,
        "candidate_count": len(scope.candidates),
        "candidate_group_count": len(scope.candidate_groups),
        "structural_boundaries": structural_boundaries,
        "skipped": scope.skipped,
        "skip_reason": scope.skip_reason,
        "call_budget_exhausted": scope.call_budget_exhausted,
        "response_status": response_diagnostics.response_status if response_diagnostics else None,
        "incomplete_reason": response_diagnostics.incomplete_reason if response_diagnostics else None,
        "response_max_output_tokens": response_diagnostics.response_max_output_tokens if response_diagnostics else None,
        "input_tokens": response_diagnostics.input_tokens if response_diagnostics else None,
        "output_tokens": response_diagnostics.output_tokens if response_diagnostics else None,
        "total_tokens": response_diagnostics.total_tokens if response_diagnostics else None,
        "reasoning_tokens": response_diagnostics.reasoning_tokens if response_diagnostics else None,
        "raw_output_chars": raw_output_chars
        if raw_output_chars is not None
        else (response_diagnostics.raw_output_chars if response_diagnostics else None),
        "output_item_statuses": response_diagnostics.output_item_statuses if response_diagnostics else [],
        "failure_reason": failure_reason,
        "elapsed_seconds": elapsed_seconds,
    }


def candidate_discovery_diagnostics(
    document: CanonicalDocumentInput,
    candidates: Sequence[StructuralCandidate],
    scopes: Sequence[LocalAnalysisScope],
) -> Dict[str, Any]:
    by_signal: Dict[str, int] = {}
    for candidate in candidates:
        for signal in candidate.evidence_signals:
            key = signal.split(":", 1)[0]
            by_signal[key] = by_signal.get(key, 0) + 1
    retained = sum(len(scope.candidates) for scope in scopes if not scope.skipped)
    candidate_calls = sum(1 for scope in scopes if scope.scope_type == "candidate_batch" and not scope.skipped)
    return {
        "total_candidates": len(candidates),
        "candidates_by_evidence_type": by_signal,
        "candidate_density_per_block": round(len(candidates) / max(1, len(document.blocks)), 4),
        "candidates_retained_for_model_analysis": retained,
        "candidate_analysis_calls": candidate_calls,
        "max_candidates_in_one_call": max((len(scope.candidates) for scope in scopes), default=0),
        "call_budget_exhausted": any(scope.call_budget_exhausted for scope in scopes),
        "skipped_candidate_scopes": sum(1 for scope in scopes if scope.skipped),
    }


def logical_nodes_from_local_structure(local: LocalRegionStructure) -> List[LogicalStructuralNode]:
    nodes: List[LogicalStructuralNode] = []
    for section in local.sections:
        candidate_ids = list(section.component_candidate_ids)
        if not candidate_ids and section.candidate_id:
            candidate_ids = [section.candidate_id]
        primary_candidate_id = section.candidate_id or (candidate_ids[0] if candidate_ids else "")
        nodes.append(
            LogicalStructuralNode(
                node_id=f"logical-{local.region_id}-{section.source_order:04d}",
                candidate_ids=candidate_ids,
                primary_candidate_id=primary_candidate_id,
                source_title=section.source_title,
                display_title=section.semantic_title,
                level=section.level,
                parent_title=section.parent_title,
                expected_start=section.expected_start,
                expected_end=section.expected_end,
                confidence=section.confidence,
                structural_role=section.structural_role,
                source_order=section.source_order,
                region_id=local.region_id,
                provenance={
                    "candidate_id": section.candidate_id,
                    "group_id": section.group_id,
                    "component_candidate_ids": candidate_ids,
                    "primary_candidate_id": primary_candidate_id,
                    "classification": section.classification,
                },
            )
        )
    return nodes


def _logical_node_classification(node: LogicalStructuralNode) -> str:
    return str(node.provenance.get("classification") or LOCAL_CLASS_STRUCTURAL)


def _logical_node_group_id(node: LogicalStructuralNode) -> Optional[str]:
    value = node.provenance.get("group_id")
    return str(value) if value else None


def reconcile_hierarchy(
    document: CanonicalDocumentInput,
    reconnaissance: DocumentReconnaissance,
    local_structures: Sequence[LocalRegionStructure],
) -> CanonicalDocumentStructure:
    sections: List[CanonicalSection] = []
    diagnostics: Dict[str, Any] = {
        "duplicate_canonical_paths": 0,
        "self_parent_relationships": 0,
        "impossible_parent_order": 0,
        "continuation_nodes": 0,
        "local_only_nodes": 0,
        "logical_nodes": 0,
        "multi_component_logical_nodes": 0,
        "macro_local_duplicate_nodes_suppressed": 0,
        "primary_g2_anchor_nodes": 0,
        "overlapping_region_boundaries": 0,
        "unresolved_model_only_sections": 0,
        "notes": [],
    }
    path_keys: set[str] = set()
    title_to_id: Dict[Tuple[str, str], str] = {}
    source_order = 0

    region_by_id = {region.region_id: region for region in reconnaissance.macro_regions}
    locals_by_region: Dict[str, List[LocalRegionStructure]] = {}
    for local in local_structures:
        locals_by_region.setdefault(local.region_id, []).append(local)
    for region in reconnaissance.macro_regions:
        region_locals = locals_by_region.get(region.region_id, [])
        if not region_locals:
            continue
        source_order += 1
        region_section_id = f"section-{source_order:04d}"
        sections.append(
            CanonicalSection(
                section_id=region_section_id,
                source_title=region.source_title,
                semantic_title=region.semantic_title,
                level=1,
                parent_id=None,
                parent_title=None,
                region_id=region.region_id,
                expected_start=region.approximate_start,
                expected_end=region.approximate_end,
                confidence=region.confidence,
                structural_role=STRUCTURAL_ROLE_MACRO,
                source_order=source_order,
            )
        )
        title_to_id[(region.region_id, _title_key(region.semantic_title))] = region_section_id
        if region.source_title:
            title_to_id[(region.region_id, _title_key(region.source_title))] = region_section_id
        macro_boundary_key = _position_boundary_key(region.approximate_start) if region.approximate_start else None
        macro_title_keys = {
            _title_key(title)
            for title in (region.source_title, region.semantic_title)
            if title
        }

        for local in sorted(
            region_locals,
            key=lambda item: (item.diagnostics or {}).get("subscope_index", 1),
        ):
            for logical_node in logical_nodes_from_local_structure(local):
                diagnostics["logical_nodes"] += 1
                if len(logical_node.candidate_ids) > 1:
                    diagnostics["multi_component_logical_nodes"] += 1
                if logical_node.primary_candidate_id:
                    diagnostics["primary_g2_anchor_nodes"] += 1
                if logical_node.structural_role not in SEGMENTING_ROLES:
                    if logical_node.structural_role == STRUCTURAL_ROLE_CONTINUATION:
                        diagnostics["continuation_nodes"] += 1
                    else:
                        diagnostics["local_only_nodes"] += 1
                    continue
                if _logical_node_classification(logical_node) in {LOCAL_CLASS_LOCAL, LOCAL_CLASS_NON_STRUCTURAL}:
                    diagnostics["local_only_nodes"] += 1
                    continue
                if not logical_node.source_title:
                    diagnostics["unresolved_model_only_sections"] += 1
                    continue
                if (
                    logical_node.expected_start is not None
                    and macro_boundary_key is not None
                    and _position_boundary_key(logical_node.expected_start) == macro_boundary_key
                    and _title_key(logical_node.source_title or logical_node.display_title) in macro_title_keys
                ):
                    diagnostics["macro_local_duplicate_nodes_suppressed"] += 1
                    diagnostics["continuation_nodes"] += 1
                    continue

                parent_id = None
                parent_title = logical_node.parent_title
                if parent_title:
                    parent_id = title_to_id.get((region.region_id, _title_key(parent_title)))
                if parent_id is None and logical_node.level > 1:
                    parent_id = region_section_id
                    parent_title = region.semantic_title
                if parent_id is None:
                    parent_id = region_section_id
                    parent_title = region.semantic_title
                if parent_id == f"section-{source_order + 1:04d}":
                    diagnostics["self_parent_relationships"] += 1
                    parent_id = region_section_id
                source_order += 1
                section_id = f"section-{source_order:04d}"
                if parent_id:
                    parent_section = next((item for item in sections if item.section_id == parent_id), None)
                    if parent_section and parent_section.source_order >= source_order:
                        diagnostics["impossible_parent_order"] += 1
                        parent_id = region_section_id
                path_key = _canonical_path_key(
                    sections,
                    parent_id,
                    logical_node.source_title or logical_node.display_title or "",
                    logical_node.expected_start,
                )
                if path_key in path_keys:
                    diagnostics["duplicate_canonical_paths"] += 1
                    diagnostics["continuation_nodes"] += 1
                    continue
                path_keys.add(path_key)
                canonical = CanonicalSection(
                    section_id=section_id,
                    source_title=logical_node.source_title,
                    semantic_title=logical_node.display_title or logical_node.source_title or "",
                    level=max(2, logical_node.level + 1),
                    parent_id=parent_id,
                    parent_title=parent_title,
                    region_id=region.region_id,
                    expected_start=logical_node.expected_start,
                    expected_end=logical_node.expected_end,
                    confidence=logical_node.confidence,
                    structural_role=logical_node.structural_role,
                    source_order=source_order,
                )
                if canonical.expected_start is None:
                    diagnostics["unresolved_model_only_sections"] += 1
                sections.append(canonical)
                for title in (canonical.source_title, canonical.semantic_title):
                    if title:
                        title_to_id[(region.region_id, _title_key(title))] = section_id
            diagnostics["continuation_nodes"] += len(local.continuation_nodes)
            diagnostics["local_only_nodes"] += len(local.local_only_nodes)
            if local.interpretation_failed:
                diagnostics["notes"].append(f"Local analysis failed for {region.region_id}")

    diagnostics["overlapping_region_boundaries"] = _count_overlapping_regions(region_by_id.values())
    quality = compute_structure_quality(reconnaissance, local_structures, sections, diagnostics)
    return CanonicalDocumentStructure(
        document_title=reconnaissance.document_title or document.title,
        document_type=reconnaissance.document_type or document.file_type,
        structure_quality=quality,
        macro_regions=list(reconnaissance.macro_regions),
        sections=sections,
        diagnostics=diagnostics,
    )


def compute_structure_quality(
    reconnaissance: DocumentReconnaissance,
    local_structures: Sequence[LocalRegionStructure],
    sections: Sequence[CanonicalSection],
    diagnostics: Dict[str, Any],
) -> StructureQuality:
    score = 0.35
    reasons = []
    if confidence_rank(reconnaissance.structure_confidence) >= 3:
        score += 0.22
        reasons.append("high-confidence global reconnaissance")
    elif confidence_rank(reconnaissance.structure_confidence) == 2:
        score += 0.12
        reasons.append("medium-confidence global reconnaissance")
    else:
        reasons.append("low-confidence global reconnaissance")
    if reconnaissance.macro_regions:
        score += 0.16
        reasons.append("macro-regions identified")
    if any(local.sections for local in local_structures):
        score += 0.14
        reasons.append("local structural analysis produced sections")
    if diagnostics.get("duplicate_canonical_paths"):
        score -= 0.16
        reasons.append("duplicate canonical paths detected")
    if diagnostics.get("impossible_parent_order") or diagnostics.get("self_parent_relationships"):
        score -= 0.18
        reasons.append("invalid parent relationships detected")
    if diagnostics.get("overlapping_region_boundaries"):
        score -= 0.12
        reasons.append("overlapping macro-region boundaries detected")
    if any(local.interpretation_failed for local in local_structures):
        score -= 0.32
        reasons.append("one or more local analyses failed")
    if not sections:
        score -= 0.2
        reasons.append("no canonical sections materialized")
    score = round(max(0.0, min(1.0, score)), 3)
    if score >= 0.78:
        status = STRUCTURE_STATUS_EXCELLENT
    elif score >= 0.5:
        status = STRUCTURE_STATUS_USABLE
    else:
        status = STRUCTURE_STATUS_WEAK
    return StructureQuality(score=score, status=status, reasons=reasons)


def compute_mapping_fidelity(mapping_result: DocumentStructureMappingResult) -> MappingFidelity:
    score = mapping_result.mapping_confidence
    reasons = []
    if mapping_result.mapping_status == STATUS_ACCEPTED:
        reasons.append("all canonical boundaries anchored with high confidence")
        status = MAPPING_STATUS_HIGH
    elif mapping_result.mapping_status == STATUS_PARTIAL:
        reasons.append("some canonical boundaries anchored safely")
        status = MAPPING_STATUS_PARTIAL
    else:
        reasons.append("canonical boundaries could not be grounded reliably")
        status = MAPPING_STATUS_LOW
    if mapping_result.unresolved_sections:
        reasons.append(f"{mapping_result.unresolved_sections} unresolved sections")
    if mapping_result.duplicate_anchor_collisions:
        reasons.append("duplicate anchor collisions detected")
    if mapping_result.source_order_violations:
        reasons.append("source-order violations detected")
    return MappingFidelity(score=score, status=status, reasons=reasons)


def strict_schema_violations(schema: Dict[str, Any]) -> List[str]:
    violations: List[str] = []

    def walk(node: Dict[str, Any], path: str) -> None:
        schema_type = node.get("type")
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        if "object" in types:
            properties = node.get("properties", {})
            required = set(node.get("required", []))
            if set(properties.keys()) != required:
                violations.append(f"{path}: required keys do not match properties")
            if node.get("additionalProperties", True) is not False:
                violations.append(f"{path}: additionalProperties must be false")
        for key, child in node.get("properties", {}).items():
            if isinstance(child, dict):
                walk(child, f"{path}.properties.{key}")
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, f"{path}.items")
        for keyword in ("anyOf", "oneOf", "allOf"):
            for index, child in enumerate(node.get(keyword, []) or []):
                if isinstance(child, dict):
                    walk(child, f"{path}.{keyword}[{index}]")

    walk(schema, "schema")
    return violations


def _ground_single_region(
    blocks: Sequence[CanonicalBlock],
    region: MacroRegion,
    occupied: set[Tuple[str, int, int, int, int]],
) -> MacroRegionGrounding:
    if region.approximate_start:
        index = _nearest_block_index(blocks, region.approximate_start)
        if index is not None:
            position = region.approximate_start
            if _position_key(position) in occupied:
                return _unresolved_grounding(region, "duplicate approximate_start anchor")
            return MacroRegionGrounding(
                region_id=region.region_id,
                source_title=region.source_title,
                semantic_title=region.semantic_title,
                source_position=position,
                block_list_index=index,
                method="approximate_start",
                confidence=1.0,
                fidelity="explicit",
                status="grounded",
            )

    if not region.source_title:
        return _unresolved_grounding(region, "source_title is null")

    for method in ("exact", "whitespace", "unicode", "punctuation"):
        for index, block in enumerate(blocks):
            match = _match_title_in_block(region.source_title, block, method)
            if not match:
                continue
            position = SourcePosition(
                unit_type=block.source_position.unit_type,
                unit_index=block.source_position.unit_index,
                block_index=block.source_position.block_index,
                start_offset=match[0],
                end_offset=match[1],
            )
            if _position_key(position) in occupied:
                return _unresolved_grounding(region, "duplicate source_title anchor")
            return MacroRegionGrounding(
                region_id=region.region_id,
                source_title=region.source_title,
                semantic_title=region.semantic_title,
                source_position=position,
                block_list_index=index,
                method=f"{method}_source_title",
                confidence=1.0 if method == "exact" else 0.92,
                fidelity="exact" if method == "exact" else "normalized",
                status="grounded",
            )

    best = None
    for index, block in enumerate(blocks):
        match = _fuzzy_title_match(region.source_title, block)
        if match and (best is None or match[2] > best[2]):
            best = (index, block, *match)
    if best:
        index, block, start, end, confidence = best
        position = SourcePosition(
            unit_type=block.source_position.unit_type,
            unit_index=block.source_position.unit_index,
            block_index=block.source_position.block_index,
            start_offset=start,
            end_offset=end,
        )
        if _position_key(position) in occupied:
            return _unresolved_grounding(region, "duplicate fuzzy source_title anchor")
        return MacroRegionGrounding(
            region_id=region.region_id,
            source_title=region.source_title,
            semantic_title=region.semantic_title,
            source_position=position,
            block_list_index=index,
            method="fuzzy_source_title",
            confidence=round(confidence, 3),
            fidelity="fuzzy",
            status="grounded",
        )

    return _unresolved_grounding(region, "source_title did not match canonical document")


def _unresolved_grounding(region: MacroRegion, reason: str) -> MacroRegionGrounding:
    return MacroRegionGrounding(
        region_id=region.region_id,
        source_title=region.source_title,
        semantic_title=region.semantic_title,
        source_position=None,
        block_list_index=None,
        method="unresolved",
        confidence=0.0,
        fidelity="unresolved",
        status="unresolved",
        reason=reason,
    )


def _slice_blocks_between(
    blocks: Sequence[CanonicalBlock],
    *,
    start_position: SourcePosition,
    end_position: SourcePosition,
) -> List[CanonicalBlock]:
    sliced: List[CanonicalBlock] = []
    for block in blocks:
        source = block.source_position
        if _block_before_position(source, start_position) or _block_after_position(source, end_position):
            continue
        raw_text = block.raw_text or block.text or ""
        start_offset = 0
        end_offset = len(raw_text)
        if _same_source_block(source, start_position):
            start_offset = max(0, min(start_position.start_offset, len(raw_text)))
        if _same_source_block(source, end_position):
            end_offset = max(start_offset, min(end_position.start_offset, len(raw_text)))
        if end_offset <= start_offset:
            continue
        sliced_text = raw_text[start_offset:end_offset]
        sliced.append(
            replace(
                block,
                text=sliced_text,
                raw_text=sliced_text,
                source_position=replace(
                    block.source_position,
                    start_offset=start_offset,
                    end_offset=end_offset,
                ),
                metadata={
                    **block.metadata,
                    "slice_start_offset": start_offset,
                    "slice_end_offset": end_offset,
                    "owned_source_slice": True,
                },
            )
        )
    return sliced


def _subdivide_owned_blocks(
    region: MacroRegion,
    owned_blocks: Sequence[CanonicalBlock],
    *,
    base_scope_id: str,
    total_blocks: int,
) -> List[LocalAnalysisScope]:
    blocks = list(owned_blocks)
    if not blocks:
        return [
            LocalAnalysisScope(
                scope_id=f"{base_scope_id}/unresolved-empty-range",
                scope_type="grounded_region",
                region=region,
                blocks=[],
                start_unit="n/a",
                end_unit="n/a",
                block_count=0,
                input_char_count=0,
                skipped=True,
                skip_reason="grounded macro-region produced an empty physical range",
                macro_region_id=region.region_id,
                subdivision_reason="empty_grounded_range",
            )
        ]
    if _blocks_fit_local_budget(blocks):
        return [
            _apply_local_scope_guard(
                _make_scope(
                    base_scope_id,
                    "grounded_region",
                    region,
                    blocks,
                    macro_region_id=region.region_id,
                    subscope_index=1,
                    subdivision_reason="fits_local_budget",
                ),
                total_blocks=total_blocks,
            )
        ]

    chunks = _split_blocks_by_budget(blocks)
    scopes = []
    for index, chunk in enumerate(chunks, start=1):
        scope = _make_scope(
            f"{base_scope_id}/scope-{index:02d}",
            "grounded_subscope",
            region,
            chunk,
            macro_region_id=region.region_id,
            subscope_index=index,
            subdivision_reason="local_scope_budget_exceeded",
        )
        scopes.append(_apply_local_scope_guard(scope, total_blocks=total_blocks))
    return scopes


def _split_blocks_by_budget(blocks: Sequence[CanonicalBlock]) -> List[List[CanonicalBlock]]:
    chunks: List[List[CanonicalBlock]] = []
    current: List[CanonicalBlock] = []
    current_chars = 0
    for block in blocks:
        text_length = len(block.raw_text or block.text or "")
        would_exceed = (
            current
            and (
                len(current) >= MAX_LOCAL_ANALYSIS_BLOCKS
                or current_chars + text_length > MAX_LOCAL_ANALYSIS_CHARS
            )
        )
        if would_exceed:
            chunks.append(current)
            if SUBSCOPE_OVERLAP_BLOCKS > 0:
                current = current[-SUBSCOPE_OVERLAP_BLOCKS:]
                current_chars = sum(len(item.raw_text or item.text or "") for item in current)
            else:
                current = []
                current_chars = 0
        if text_length > MAX_LOCAL_ANALYSIS_CHARS:
            chunks.extend(_split_single_large_block(block))
            current = []
            current_chars = 0
            continue
        current.append(block)
        current_chars += text_length
    if current:
        chunks.append(current)
    return chunks


def _split_single_large_block(block: CanonicalBlock) -> List[List[CanonicalBlock]]:
    raw_text = block.raw_text or block.text or ""
    slices: List[List[CanonicalBlock]] = []
    cursor = 0
    while cursor < len(raw_text):
        end = min(len(raw_text), cursor + MAX_LOCAL_ANALYSIS_CHARS)
        sliced_text = raw_text[cursor:end]
        slices.append([
            replace(
                block,
                text=sliced_text,
                raw_text=sliced_text,
                source_position=replace(
                    block.source_position,
                    start_offset=(block.source_position.start_offset or 0) + cursor,
                    end_offset=(block.source_position.start_offset or 0) + end,
                ),
                metadata={
                    **block.metadata,
                    "slice_start_offset": (block.source_position.start_offset or 0) + cursor,
                    "slice_end_offset": (block.source_position.start_offset or 0) + end,
                    "owned_source_slice": True,
                    "deterministic_text_window": True,
                },
            )
        ])
        cursor = end
    return slices


def _make_scope(
    scope_id: str,
    scope_type: str,
    region: MacroRegion,
    blocks: Sequence[CanonicalBlock],
    *,
    macro_region_id: Optional[str] = None,
    subscope_index: int = 1,
    subdivision_reason: str = "none",
) -> LocalAnalysisScope:
    chosen = list(blocks)
    input_chars = sum(len(block.raw_text or block.text or "") for block in chosen)
    source_start = chosen[0].source_position if chosen else None
    source_end = chosen[-1].source_position if chosen else None
    return LocalAnalysisScope(
        scope_id=scope_id,
        scope_type=scope_type,
        region=region,
        blocks=chosen,
        start_unit=_unit_label(chosen[0]) if chosen else "n/a",
        end_unit=_unit_label(chosen[-1]) if chosen else "n/a",
        block_count=len(chosen),
        input_char_count=input_chars,
        macro_region_id=macro_region_id or region.region_id,
        subscope_index=subscope_index,
        subdivision_reason=subdivision_reason,
        source_start=source_start,
        source_end=source_end,
    )


def _apply_local_scope_guard(scope: LocalAnalysisScope, *, total_blocks: int) -> LocalAnalysisScope:
    repeated_whole_document = (
        scope.scope_type == "grounded_region"
        and scope.block_count == total_blocks
        and total_blocks > SMALL_DOCUMENT_BLOCK_LIMIT
    )
    oversized = (
        scope.block_count > MAX_LOCAL_ANALYSIS_BLOCKS
        or scope.input_char_count > MAX_LOCAL_ANALYSIS_CHARS
    )
    if not repeated_whole_document and not oversized:
        return scope
    reasons = []
    if repeated_whole_document:
        reasons.append("grounded region would pass the entire large document")
    if oversized:
        reasons.append("local analysis scope exceeds configured safety limits")
    return LocalAnalysisScope(
        scope_id=scope.scope_id,
        scope_type=scope.scope_type,
        region=scope.region,
        blocks=[],
        start_unit=scope.start_unit,
        end_unit=scope.end_unit,
        block_count=scope.block_count,
        input_char_count=scope.input_char_count,
        skipped=True,
        skip_reason="; ".join(reasons),
        macro_region_id=scope.macro_region_id,
        subscope_index=scope.subscope_index,
        subdivision_reason=scope.subdivision_reason,
        source_start=scope.source_start,
        source_end=scope.source_end,
    )


def _blocks_fit_local_budget(blocks: Sequence[CanonicalBlock]) -> bool:
    return (
        len(blocks) <= MAX_LOCAL_ANALYSIS_BLOCKS
        and sum(len(block.raw_text or block.text or "") for block in blocks)
        <= MAX_LOCAL_ANALYSIS_CHARS
    )


def _is_small_document(blocks: Sequence[CanonicalBlock]) -> bool:
    return (
        len(blocks) <= SMALL_DOCUMENT_BLOCK_LIMIT
        and sum(len(block.raw_text or block.text or "") for block in blocks) <= MAX_LOCAL_ANALYSIS_CHARS
    )


def _block_before_position(block_position: SourcePosition, position: SourcePosition) -> bool:
    return (
        block_position.unit_type,
        block_position.unit_index,
        block_position.block_index,
    ) < (
        position.unit_type,
        position.unit_index,
        position.block_index,
    )


def _block_after_position(block_position: SourcePosition, position: SourcePosition) -> bool:
    return (
        block_position.unit_type,
        block_position.unit_index,
        block_position.block_index,
    ) > (
        position.unit_type,
        position.unit_index,
        position.block_index,
    )


def _same_source_block(first: SourcePosition, second: SourcePosition) -> bool:
    return (
        first.unit_type == second.unit_type
        and first.unit_index == second.unit_index
        and first.block_index == second.block_index
    )


def _block_end_position(block: CanonicalBlock) -> SourcePosition:
    raw_text = block.raw_text or block.text or ""
    return replace(
        block.source_position,
        start_offset=len(raw_text),
        end_offset=len(raw_text),
    )


def _match_title_in_block(title: str, block: CanonicalBlock, method: str) -> Optional[Tuple[int, int]]:
    text = block.raw_text or block.text
    if method == "exact":
        offset = text.find(title)
        return (offset, offset + len(title)) if offset >= 0 else None
    normalized_text, text_map = _normalized_with_map(text, method)
    normalized_title, _ = _normalized_with_map(title, method)
    if not normalized_title:
        return None
    offset = normalized_text.find(normalized_title)
    if offset < 0 or not text_map:
        return None
    start = text_map[offset]
    end = text_map[min(offset + len(normalized_title) - 1, len(text_map) - 1)] + 1
    return start, end


def _fuzzy_title_match(title: str, block: CanonicalBlock) -> Optional[Tuple[int, int, float]]:
    best = None
    title_key = _compact_key(title)
    if not title_key or len(title_key) < 8:
        return None
    for start, end, line in _iter_nonempty_lines(block.raw_text or block.text):
        line_key = _compact_key(line)
        if not line_key:
            continue
        ratio = difflib.SequenceMatcher(None, title_key, line_key).ratio()
        length_ratio = min(len(title_key), len(line_key)) / max(len(title_key), len(line_key))
        if ratio >= 0.88 and length_ratio >= 0.7 and (best is None or ratio > best[2]):
            best = (start, end, ratio)
    return best


def _iter_nonempty_lines(text: str) -> Iterable[Tuple[int, int, str]]:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        start = offset + len(line) - len(line.lstrip())
        end = start + len(stripped)
        if stripped:
            yield start, end, stripped
        offset += len(raw_line)


def _normalized_with_map(text: str, method: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    mapping: List[int] = []
    previous_space = False
    for index, char in enumerate(text):
        transformed = char
        if method in {"unicode", "punctuation"}:
            transformed = unicodedata.normalize("NFKD", transformed)
            transformed = "".join(c for c in transformed if not unicodedata.combining(c))
        for inner in transformed:
            if method == "punctuation" and unicodedata.category(inner).startswith("P"):
                inner = " "
            if method in {"whitespace", "unicode", "punctuation"} and inner.isspace():
                if previous_space:
                    continue
                inner = " "
                previous_space = True
            else:
                previous_space = False
            chars.append(inner.casefold())
            mapping.append(index)
    normalized = "".join(chars).strip()
    leading = len("".join(chars)) - len("".join(chars).lstrip())
    if leading:
        mapping = mapping[leading:]
    return normalized, mapping[:len(normalized)]


def _compact_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _position_key(position: SourcePosition) -> Tuple[str, int, int, int, int]:
    return (
        position.unit_type,
        position.unit_index,
        position.block_index,
        position.start_offset,
        position.end_offset or position.start_offset,
    )


def _unit_label(block: CanonicalBlock) -> str:
    return f"{block.source_position.unit_type}:{block.source_position.unit_index}"


def _position_label(position: Optional[SourcePosition]) -> Optional[str]:
    if position is None:
        return None
    return (
        f"{position.unit_type}:{position.unit_index}:"
        f"{position.block_index}@{position.start_offset}"
        f"-{position.end_offset}"
    )


def _position_unit_label(position: Optional[SourcePosition]) -> str:
    if position is None:
        return "n/a"
    return f"{position.unit_type}:{position.unit_index}"


def _position_from_json(value: Any) -> Optional[SourcePosition]:
    if not isinstance(value, dict):
        return None
    unit_type = _optional_text(value.get("unit_type"))
    if not unit_type or value.get("unit_index") is None or value.get("block_index") is None:
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


def _nearest_block_index(
    blocks: Sequence[CanonicalBlock],
    position: Optional[SourcePosition],
) -> Optional[int]:
    if position is None:
        return None
    for index, block in enumerate(blocks):
        source = block.source_position
        if (
            source.unit_type == position.unit_type
            and source.unit_index == position.unit_index
            and source.block_index == position.block_index
        ):
            return index
    for index, block in enumerate(blocks):
        source = block.source_position
        if source.unit_type == position.unit_type and source.unit_index == position.unit_index:
            return index
    return None


def _count_overlapping_regions(regions: Iterable[MacroRegion]) -> int:
    ordered = sorted(
        [region for region in regions if region.source_start_order is not None],
        key=lambda region: region.source_start_order or 0,
    )
    overlaps = 0
    previous_end = None
    for region in ordered:
        if previous_end is not None and (region.source_start_order or 0) < previous_end:
            overlaps += 1
        previous_end = max(previous_end or 0, region.source_end_order or 0)
    return overlaps


def _canonical_path_key(
    existing: Sequence[CanonicalSection],
    parent_id: Optional[str],
    title: str,
    expected_start: Optional[SourcePosition] = None,
) -> str:
    parts = [_title_key(title)]
    lookup = {item.section_id: item for item in existing}
    current = lookup.get(parent_id or "")
    guard = 0
    while current and guard < 16:
        parts.append(_title_key(current.source_title or current.semantic_title))
        current = lookup.get(current.parent_id or "")
        guard += 1
    anchor = _position_label(expected_start) or "unanchored"
    return " > ".join(reversed([part for part in parts if part])) + f" @{anchor}"


def _confidence_from_quality(quality: StructureQuality) -> str:
    if quality.score >= 0.75:
        return "HIGH"
    if quality.score >= 0.5:
        return "MEDIUM"
    return "LOW"


def _title_key(value: Optional[str]) -> str:
    return " ".join((value or "").casefold().split())


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _required_text(value: Any, default: str) -> str:
    return _optional_text(value) or default


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = default
    return max(minimum, min(maximum, integer))


def _enum(value: Any, allowed: Tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _local_classification_from_raw(raw: Dict[str, Any]) -> str:
    text = str(raw.get("classification") or "").strip()
    if text in LOCAL_CLASSIFICATIONS:
        return text
    role = _enum(raw.get("structural_role"), STRUCTURAL_ROLES, STRUCTURAL_ROLE_SECTION)
    if role in SEGMENTING_ROLES:
        return LOCAL_CLASS_STRUCTURAL
    if role == STRUCTURAL_ROLE_CONTINUATION:
        return LOCAL_CLASS_CONTINUATION
    return LOCAL_CLASS_LOCAL
