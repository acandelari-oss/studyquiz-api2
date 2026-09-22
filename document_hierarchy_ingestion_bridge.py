"""Production ingestion bridge for preserved document hierarchy.

This module is intentionally narrow: it adapts the isolated hierarchy
preservation layer into the existing DOUNO upload contract by enriching
ExtractedBlock.section.  It does not change chunking, embeddings, topic
generation, persistence schema, or downstream learning features.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from canonical_document import adapt_document
from document_extractors import ExtractedBlock, ExtractedDocument
from document_hierarchy_contracts import (
    CanonicalDocumentNode,
    CanonicalDocumentStructure,
    STATUS_ACCEPTED,
    STATUS_PARTIAL,
)
from document_hierarchy_evidence_mapper import map_document_to_structural_evidence
from document_hierarchy_preserver import preserve_document_hierarchy


@dataclass(frozen=True)
class HierarchyIngestionBridgeResult:
    document: ExtractedDocument
    structure: Optional[CanonicalDocumentStructure]
    applied: bool
    diagnostics: Dict[str, object]


def enrich_extracted_document_with_preserved_hierarchy(
    *,
    extracted_document: ExtractedDocument,
    file_bytes: bytes,
    document_id: str,
) -> HierarchyIngestionBridgeResult:
    """Apply preserved hierarchy labels to an already extracted document.

    The existing extraction result remains the source of text passed to
    chunking.  The canonical hierarchy path is used only to split blocks at
    detected section boundaries and set ExtractedBlock.section.
    """

    try:
        canonical_document = adapt_document(file_bytes, extracted_document.filename)
        evidence = map_document_to_structural_evidence(canonical_document)
        structure = preserve_document_hierarchy(
            evidence,
            document_id=document_id,
            document_title=extracted_document.filename,
            source_format=canonical_document.file_type,
        )
    except Exception as exc:
        return HierarchyIngestionBridgeResult(
            document=extracted_document,
            structure=None,
            applied=False,
            diagnostics={
                "fallback_reason": "hierarchy_preservation_failed",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )

    enriched_blocks, split_count, matched_count = apply_preserved_hierarchy_to_blocks(
        extracted_document.blocks,
        structure,
        fallback_section=extracted_document.filename,
    )
    applied = bool(matched_count and enriched_blocks)

    if not applied:
        return HierarchyIngestionBridgeResult(
            document=extracted_document,
            structure=structure,
            applied=False,
            diagnostics={
                "fallback_reason": "no_hierarchy_boundaries_matched_existing_extraction",
                "structure_status": structure.structure_status,
                "structure_confidence": structure.structure_confidence,
                "node_count": len(structure.nodes),
                "matched_boundaries": matched_count,
                "split_blocks_created": split_count,
            },
        )

    return HierarchyIngestionBridgeResult(
        document=ExtractedDocument(
            filename=extracted_document.filename,
            file_format=extracted_document.file_format,
            file_size_bytes=extracted_document.file_size_bytes,
            blocks=enriched_blocks,
            pages_detected=extracted_document.pages_detected,
        ),
        structure=structure,
        applied=True,
        diagnostics={
            "structure_status": structure.structure_status,
            "structure_confidence": structure.structure_confidence,
            "node_count": len(structure.nodes),
            "matched_boundaries": matched_count,
            "split_blocks_created": split_count,
        },
    )


def apply_preserved_hierarchy_to_blocks(
    blocks: Sequence[ExtractedBlock],
    structure: CanonicalDocumentStructure,
    *,
    fallback_section: str,
) -> Tuple[List[ExtractedBlock], int, int]:
    """Return blocks split/labeled by accepted preserved hierarchy nodes."""

    nodes = _accepted_nodes(structure)
    if not nodes:
        return list(blocks), 0, 0

    paths_by_node_id = _section_paths(nodes)
    active_section: Optional[str] = None
    used_node_ids: set[str] = set()
    output: List[ExtractedBlock] = []
    split_count = 0
    matched_count = 0

    for original_block_index, block in enumerate(blocks, start=1):
        text = block.text or ""
        raw_text = block.raw_text or text
        if not text.strip():
            output.append(block)
            continue

        candidates = _candidate_nodes_for_block(nodes, block, used_node_ids)
        boundaries = _find_boundaries(text, candidates, used_node_ids, paths_by_node_id)
        if not boundaries:
            output.append(
                ExtractedBlock(
                    text=block.text,
                    raw_text=block.raw_text,
                    page=block.page,
                    section=active_section or block.section or fallback_section,
                    block_index=len(output) + 1,
                )
            )
            continue

        matched_count += len(boundaries)
        cursor = 0
        for offset, node in boundaries:
            if offset > cursor:
                _append_split_block(
                    output,
                    text[cursor:offset],
                    raw_text[cursor:offset] if len(raw_text) == len(text) else text[cursor:offset],
                    block,
                    active_section or block.section or fallback_section,
                )
            active_section = paths_by_node_id[node.node_id]
            cursor = offset

        if cursor < len(text):
            _append_split_block(
                output,
                text[cursor:],
                raw_text[cursor:] if len(raw_text) == len(text) else text[cursor:],
                block,
                active_section or block.section or fallback_section,
            )

        if len(output) > original_block_index:
            split_count += len(boundaries)

    return output or list(blocks), split_count, matched_count


def _accepted_nodes(
    structure: CanonicalDocumentStructure,
) -> List[CanonicalDocumentNode]:
    if structure.structure_status not in {STATUS_ACCEPTED, STATUS_PARTIAL}:
        return []
    return [
        node
        for node in sorted(structure.nodes, key=lambda item: item.source_order)
        if node.status == STATUS_ACCEPTED
        and node.source_title
        and node.node_kind == "document_section"
    ]


def _section_paths(
    nodes: Sequence[CanonicalDocumentNode],
) -> Dict[str, str]:
    by_id = {node.node_id: node for node in nodes}
    paths: Dict[str, str] = {}

    def build(node: CanonicalDocumentNode) -> str:
        if node.node_id in paths:
            return paths[node.node_id]
        parts = []
        current: Optional[CanonicalDocumentNode] = node
        guard = 0
        while current is not None and guard < 16:
            title = current.source_title or current.logical_title
            if title:
                parts.append(title)
            current = by_id.get(current.parent_id) if current.parent_id else None
            guard += 1
        path = " > ".join(reversed(parts))
        paths[node.node_id] = path
        return path

    for node in nodes:
        build(node)
    return paths


def _candidate_nodes_for_block(
    nodes: Sequence[CanonicalDocumentNode],
    block: ExtractedBlock,
    used_node_ids: set[str],
) -> List[CanonicalDocumentNode]:
    candidates = []
    for node in nodes:
        if node.node_id in used_node_ids:
            continue
        if block.page is not None and node.source_span is not None:
            start = node.source_span.start
            if start.unit_type == "page" and start.unit_index != block.page:
                continue
        candidates.append(node)
    return candidates


def _find_boundaries(
    text: str,
    candidates: Sequence[CanonicalDocumentNode],
    used_node_ids: set[str],
    paths_by_node_id: Dict[str, str],
) -> List[Tuple[int, CanonicalDocumentNode]]:
    occupied_offsets: set[int] = set()
    boundaries: List[Tuple[int, CanonicalDocumentNode]] = []
    search_start = 0
    for node in candidates:
        title = node.source_title or node.logical_title or ""
        match = _find_title_offset(text, title, start=search_start)
        if match is None or match in occupied_offsets:
            continue
        occupied_offsets.add(match)
        boundaries.append((match, node))
        used_node_ids.add(node.node_id)
        search_start = match

    boundaries.sort(key=lambda item: (item[0], item[1].source_order))
    return [
        (offset, node)
        for offset, node in boundaries
        if paths_by_node_id.get(node.node_id)
    ]


def _find_title_offset(text: str, title: str, *, start: int = 0) -> Optional[int]:
    if not text or not title:
        return None
    exact = text.find(title, start)
    if exact >= 0:
        return exact

    normalized_text, text_map = _normalized_with_map(text)
    normalized_title, _ = _normalized_with_map(title)
    if not normalized_title:
        return None
    normalized_start = 0
    if start > 0 and text_map:
        normalized_start = next(
            (index for index, original in enumerate(text_map) if original >= start),
            len(text_map),
        )
    normalized_offset = normalized_text.find(normalized_title, normalized_start)
    if normalized_offset < 0 or normalized_offset >= len(text_map):
        return None
    return text_map[normalized_offset]


def _normalized_with_map(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    mapping: List[int] = []
    previous_space = False
    for index, char in enumerate(text):
        transformed = unicodedata.normalize("NFKD", char)
        transformed = "".join(c for c in transformed if not unicodedata.combining(c))
        for inner in transformed:
            if inner.isspace() or unicodedata.category(inner).startswith("P"):
                if previous_space:
                    continue
                chars.append(" ")
                mapping.append(index)
                previous_space = True
            else:
                chars.append(inner.casefold())
                mapping.append(index)
                previous_space = False
    normalized = "".join(chars)
    leading = len(normalized) - len(normalized.lstrip())
    trailing = len(normalized.rstrip())
    return normalized.strip(), mapping[leading:trailing]


def _append_split_block(
    output: List[ExtractedBlock],
    text: str,
    raw_text: str,
    source_block: ExtractedBlock,
    section: str,
) -> None:
    if not text or not text.strip():
        return
    output.append(
        ExtractedBlock(
            text=text.strip(),
            raw_text=(raw_text or text).strip(),
            page=source_block.page,
            section=section,
            block_index=len(output) + 1,
        )
    )
