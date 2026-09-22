"""Deterministic anchoring for model-native document structure.

This prototype maps model-proposed headings back onto ExtractedDocument raw
text without touching ingestion or persistence.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from document_extractors import ExtractedDocument
from model_structure_interpreter import DocumentStructure, ModelSection


SOURCE_TITLE_FIDELITY_VALUES = ("exact", "normalized", "fuzzy", "model_only")


@dataclass(frozen=True)
class Anchor:
    page: Optional[int]
    block: Optional[int]
    start_offset: Optional[int]
    end_offset: Optional[int]
    method: str
    confidence: float
    source_title_fidelity: str


@dataclass(frozen=True)
class AnchoredSection(ModelSection):
    anchor: Optional[Anchor] = None


@dataclass(frozen=True)
class AnchoredDocumentStructure:
    document_title: str
    document_type: str
    confidence: str
    analyzer: str
    analyzer_version: str
    sections: List[AnchoredSection]
    local_structure_summary: Dict
    notes: List[str]


@dataclass(frozen=True)
class AnchorSummary:
    anchored: int
    unresolved: int
    exact: int
    normalized: int
    fuzzy: int
    model_only: int
    duplicate_anchor_collisions: int
    source_order_violations: int


def anchor_document_structure(
    structure: DocumentStructure,
    extracted_document: ExtractedDocument,
) -> tuple[AnchoredDocumentStructure, AnchorSummary]:
    block_indexes = _build_block_indexes(extracted_document)
    occupied_spans: set[tuple[int, int, int]] = set()
    anchored_sections: List[AnchoredSection] = []
    previous_global_position: Optional[Tuple[int, int, int]] = None
    duplicate_collisions = 0
    order_violations = 0

    for section in structure.sections:
        anchor = _find_anchor_for_section(
            section=section,
            block_indexes=block_indexes,
            occupied_spans=occupied_spans,
        )

        if anchor:
            span_key = (
                anchor.block or -1,
                anchor.start_offset or 0,
                anchor.end_offset or 0,
            )
            if span_key in occupied_spans:
                duplicate_collisions += 1
                anchor = None
            else:
                current_position = (
                    anchor.block or 0,
                    anchor.start_offset or 0,
                    section.source_order,
                )
                if (
                    previous_global_position is not None
                    and current_position < previous_global_position
                ):
                    order_violations += 1
                    anchor = None
                else:
                    occupied_spans.add(span_key)
                    previous_global_position = current_position

        anchored_sections.append(
            AnchoredSection(
                **{
                    field: getattr(section, field)
                    for field in ModelSection.__dataclass_fields__
                },
                anchor=anchor,
            )
        )

    counts = {
        "exact": 0,
        "normalized": 0,
        "fuzzy": 0,
        "model_only": 0,
    }
    for section in anchored_sections:
        if section.anchor:
            counts[section.anchor.source_title_fidelity] += 1
        else:
            counts["model_only"] += 1

    summary = AnchorSummary(
        anchored=sum(1 for section in anchored_sections if section.anchor),
        unresolved=sum(1 for section in anchored_sections if not section.anchor),
        exact=counts["exact"],
        normalized=counts["normalized"],
        fuzzy=counts["fuzzy"],
        model_only=counts["model_only"],
        duplicate_anchor_collisions=duplicate_collisions,
        source_order_violations=order_violations,
    )

    return (
        AnchoredDocumentStructure(
            document_title=structure.document_title,
            document_type=structure.document_type,
            confidence=structure.confidence,
            analyzer=structure.analyzer,
            analyzer_version=structure.analyzer_version,
            sections=anchored_sections,
            local_structure_summary=structure.local_structure_summary,
            notes=structure.notes,
        ),
        summary,
    )


def _find_anchor_for_section(
    *,
    section: ModelSection,
    block_indexes: List[Dict],
    occupied_spans: set[tuple[int, int, int]],
) -> Optional[Anchor]:
    if not section.source_title:
        return None

    candidate_blocks = _candidate_blocks(section, block_indexes)
    title = section.source_title

    for method in ("exact", "whitespace", "unicode", "punctuation"):
        for block in candidate_blocks:
            match = _match_in_block(title, block, method=method)
            if match and (block["block"], match[0], match[1]) not in occupied_spans:
                fidelity = "exact" if method == "exact" else "normalized"
                confidence = 1.0 if method == "exact" else 0.92
                return Anchor(
                    page=block["page"],
                    block=block["block"],
                    start_offset=match[0],
                    end_offset=match[1],
                    method=method,
                    confidence=confidence,
                    source_title_fidelity=fidelity,
                )

    for block in candidate_blocks:
        match = _fuzzy_line_match(title, block)
        if match and (block["block"], match[0], match[1]) not in occupied_spans:
            return Anchor(
                page=block["page"],
                block=block["block"],
                start_offset=match[0],
                end_offset=match[1],
                method="page_constrained_fuzzy",
                confidence=match[2],
                source_title_fidelity="fuzzy",
            )

    return None


def _candidate_blocks(section: ModelSection, block_indexes: List[Dict]) -> List[Dict]:
    if section.start_page is not None:
        exact = [block for block in block_indexes if block["page"] == section.start_page]
        if exact:
            return exact

        nearby = [
            block
            for block in block_indexes
            if block["page"] is not None
            and abs(int(block["page"]) - int(section.start_page)) <= 1
        ]
        if nearby:
            return nearby

    return block_indexes


def _build_block_indexes(extracted_document: ExtractedDocument) -> List[Dict]:
    return [
        {
            "page": block.page,
            "block": block.block_index or index,
            "text": block.raw_text or block.text or "",
        }
        for index, block in enumerate(extracted_document.blocks, start=1)
    ]


def _match_in_block(title: str, block: Dict, *, method: str) -> Optional[Tuple[int, int]]:
    text = block["text"]
    if method == "exact":
        offset = text.find(title)
        return (offset, offset + len(title)) if offset >= 0 else None

    normalized_text, text_map = _normalized_with_map(text, method)
    normalized_title, _ = _normalized_with_map(title, method)
    if not normalized_title:
        return None
    normalized_offset = normalized_text.find(normalized_title)
    if normalized_offset < 0:
        return None
    start = text_map[normalized_offset]
    end_index = normalized_offset + len(normalized_title) - 1
    end = text_map[min(end_index, len(text_map) - 1)] + 1
    return start, end


def _fuzzy_line_match(title: str, block: Dict) -> Optional[Tuple[int, int, float]]:
    best: Optional[Tuple[int, int, float]] = None
    normalized_title = _compact_key(title)
    if not normalized_title:
        return None

    for start, end, line in _iter_nonempty_lines(block["text"]):
        candidate = _compact_key(line)
        if not candidate:
            continue
        ratio = difflib.SequenceMatcher(None, normalized_title, candidate).ratio()
        if ratio >= 0.86 and (best is None or ratio > best[2]):
            best = (start, end, ratio)
    return best


def _iter_nonempty_lines(text: str) -> Iterable[Tuple[int, int, str]]:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_without_newline = raw_line.rstrip("\r\n")
        stripped = line_without_newline.strip()
        line_start = offset + len(line_without_newline) - len(line_without_newline.lstrip())
        line_end = line_start + len(stripped)
        if stripped:
            yield line_start, line_end, stripped
        offset += len(raw_line)


def _normalized_with_map(text: str, method: str) -> Tuple[str, List[int]]:
    normalized_chars: List[str] = []
    index_map: List[int] = []
    last_was_space = False

    for index, char in enumerate(text):
        transformed = char
        if method in {"unicode", "punctuation"}:
            transformed = unicodedata.normalize("NFKD", transformed)
            transformed = "".join(c for c in transformed if not unicodedata.combining(c))
        for inner in transformed:
            if method == "punctuation" and unicodedata.category(inner).startswith("P"):
                inner = " "
            if method in {"whitespace", "unicode", "punctuation"} and inner.isspace():
                if last_was_space:
                    continue
                inner = " "
                last_was_space = True
            else:
                last_was_space = False
            normalized_chars.append(inner.casefold())
            index_map.append(index)

    normalized = "".join(normalized_chars).strip()
    leading_trim = len("".join(normalized_chars)) - len("".join(normalized_chars).lstrip())
    if leading_trim:
        index_map = index_map[leading_trim:]
    if len(index_map) > len(normalized):
        index_map = index_map[: len(normalized)]
    return normalized, index_map


def _compact_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized
