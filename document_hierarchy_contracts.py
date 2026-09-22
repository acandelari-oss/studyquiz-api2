"""Foundational contracts for document hierarchy preservation.

This module is intentionally isolated from production ingestion.  It defines
the universal evidence and final-structure contracts that later preservation
and repair layers can consume without changing DOUNO's existing downstream
features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from canonical_document import SourcePosition


SOURCE_FORMAT_PDF = "pdf"
SOURCE_FORMAT_DOCX = "docx"
SOURCE_FORMAT_PPTX = "pptx"

EVIDENCE_ROLE_TITLE = "title"
EVIDENCE_ROLE_HEADING = "heading"
EVIDENCE_ROLE_BODY = "body"
EVIDENCE_ROLE_LIST = "list"
EVIDENCE_ROLE_TABLE = "table"
EVIDENCE_ROLE_VISUAL_HEADING = "visual_heading"
EVIDENCE_ROLE_TOC_ENTRY = "toc_entry"
EVIDENCE_ROLE_UNKNOWN = "unknown"

CONTENT_KIND_DOCUMENT_BOUNDARY = "document_boundary_evidence"
CONTENT_KIND_LOCAL_STRUCTURE = "local_content_structure"
CONTENT_KIND_BODY = "body_content"
CONTENT_KIND_TABLE = "table_content"
CONTENT_KIND_FIGURE_CAPTION = "figure_caption"
CONTENT_KIND_LIST_ITEM = "list_item"
CONTENT_KIND_TOC_ENTRY = "toc_entry"
CONTENT_KIND_UNKNOWN = "unknown"

ORIGIN_EXPLICIT_SOURCE = "explicit_source"
ORIGIN_DETERMINISTIC_REPAIR = "deterministic_repair"
ORIGIN_MODEL_REPAIR = "model_repair"
ORIGIN_UNRESOLVED = "unresolved"

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

STATUS_ACCEPTED = "accepted"
STATUS_PARTIAL = "partial"
STATUS_UNRESOLVED = "unresolved"
STATUS_REJECTED = "rejected"

NODE_KIND_DOCUMENT_SECTION = "document_section"
NODE_KIND_LOCAL_STRUCTURE = "local_structure"
NODE_KIND_UNRESOLVED = "unresolved_structure"


@dataclass(frozen=True)
class SourceSpan:
    """A physical source range, not only an anchor point."""

    start: SourcePosition
    end: Optional[SourcePosition] = None


@dataclass(frozen=True)
class StructuralEvidence:
    """Format-independent source evidence before hierarchy decisions."""

    evidence_id: str
    source_format: str
    source_order: int
    source_span: SourceSpan
    raw_text: str
    normalized_text: str
    evidence_role: str = EVIDENCE_ROLE_UNKNOWN
    content_kind: str = CONTENT_KIND_BODY
    native_evidence: Dict[str, Any] = field(default_factory=dict)
    visual_evidence: Dict[str, Any] = field(default_factory=dict)
    numbering_evidence: Dict[str, Any] = field(default_factory=dict)
    structural_context: Dict[str, Any] = field(default_factory=dict)
    confidence_hint: str = CONFIDENCE_LOW
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceProvenance:
    """Minimal provenance for a final structural decision."""

    supporting_evidence_ids: List[str] = field(default_factory=list)
    conflicting_evidence_ids: List[str] = field(default_factory=list)
    decision_source: str = ORIGIN_UNRESOLVED
    repair_reason: Optional[str] = None


@dataclass(frozen=True)
class CanonicalDocumentNode:
    """A final hierarchy node produced after preservation/repair."""

    node_id: str
    source_title: Optional[str]
    logical_title: Optional[str]
    level: Optional[int]
    parent_id: Optional[str]
    source_order: int
    source_span: Optional[SourceSpan]
    origin: str
    confidence: str
    status: str
    evidence_ids: List[str] = field(default_factory=list)
    provenance: EvidenceProvenance = field(default_factory=EvidenceProvenance)
    node_kind: str = NODE_KIND_DOCUMENT_SECTION


@dataclass(frozen=True)
class CanonicalDocumentStructure:
    """Final universal hierarchy contract for a source document."""

    document_id: str
    source_format: str
    document_title: str
    structure_status: str
    structure_confidence: str
    nodes: List[CanonicalDocumentNode] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
