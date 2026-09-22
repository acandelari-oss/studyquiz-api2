"""PDF adapter for the clean DocumentInterpreterResult contract.

This isolated adapter is intentionally conservative.  It preserves PDF page
and fragment evidence, creates sections only when explicit numbering or clear
visual/editorial signals are present, and falls back instead of inventing a
model-generated organization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Set

from canonical_document import (
    ROLE_BODY,
    CanonicalBlock,
    CanonicalDocumentInput,
    adapt_pdf_document,
)
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ORIGIN_DETERMINISTIC_REPAIR,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_PDF,
    SourceSpan,
)
from document_interpreter_contract import (
    CONTENT_ROLE_BODY,
    CONTENT_ROLE_HEADING,
    DocumentInterpreterResult,
    InterpretedContentBlock,
    InterpretedDocumentSection,
    INTERPRETER_STATUS_ACCEPTED,
    INTERPRETER_STATUS_FALLBACK,
    INTERPRETER_STATUS_PARTIAL,
    validate_document_interpreter_result,
)


NUMBERED_HEADING_RE = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,4})(?P<delimiter>[\s.)-]+)(?P<title>\S.{2,140})$"
)


@dataclass(frozen=True)
class PdfVisualLine:
    line_id: str
    text: str
    raw_text: str
    source_order: int
    blocks: List[CanonicalBlock]
    page: Optional[int]
    x: Optional[float]
    y: Optional[float]
    font_size: Optional[float]
    font_name: Optional[str]


def interpret_pdf_document(
    file_bytes: bytes,
    filename: str,
    *,
    document_id: str,
) -> DocumentInterpreterResult:
    """Interpret a PDF file into conservative source sections."""

    canonical_document = adapt_pdf_document(file_bytes, filename)
    return interpret_canonical_pdf_document(
        canonical_document,
        document_id=document_id,
    )


def interpret_canonical_pdf_document(
    document: CanonicalDocumentInput,
    *,
    document_id: str,
) -> DocumentInterpreterResult:
    """Interpret an already canonicalized PDF document."""

    if document.file_type != SOURCE_FORMAT_PDF:
        return DocumentInterpreterResult(
            document_id=document_id,
            source_format=document.file_type,
            document_title=document.title or document.filename,
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[
                _content_block(block, section_id=None, role=CONTENT_ROLE_BODY)
                for block in document.blocks
                if (block.text or "").strip()
            ],
            diagnostics={
                "fallback_reason": "not_pdf",
                "source_format": document.file_type,
            },
        )

    blocks = [block for block in document.blocks if (block.text or "").strip()]
    lines = _visual_lines_from_blocks(blocks)
    body_font_size = _body_font_size_from_lines(lines)
    pdf_mode = _classify_pdf_mode(document, lines)
    if pdf_mode == "slide_export":
        return _interpret_slide_export_pdf(
            document=document,
            document_id=document_id,
            lines=lines,
            body_font_size=body_font_size,
        )

    heading_candidates = _detect_heading_candidates(lines, body_font_size)
    heading_by_line_id = {candidate["line_id"]: candidate for candidate in heading_candidates}
    diagnostics = {
        "page_count": len(document.units),
        "content_blocks": len(blocks),
        "visual_lines": len(lines),
        "heading_count": len(heading_candidates),
        "numbered_heading_count": sum(1 for item in heading_candidates if item["method"] == "numbered_heading"),
        "visual_heading_count": sum(1 for item in heading_candidates if item["method"] == "visual_heading"),
        "rejected_heading_like_count": sum(1 for item in heading_candidates if item.get("rejected")),
        "body_font_size": body_font_size,
        "pdf_mode": pdf_mode,
    }

    if not heading_candidates:
        result = DocumentInterpreterResult(
            document_id=document_id,
            source_format=SOURCE_FORMAT_PDF,
            document_title=document.title or document.filename,
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[
                _content_block_from_line(line, section_id=None, role=CONTENT_ROLE_BODY)
                for line in lines
            ],
            diagnostics={
                **diagnostics,
                "fallback_reason": "pdf_contains_no_high_confidence_headings",
            },
        )
        return _with_validation_diagnostics(result)

    sections: List[InterpretedDocumentSection] = []
    unresolved_blocks: List[InterpretedContentBlock] = []
    active_section_id: Optional[str] = None
    owned_blocks_by_section: Dict[str, List[InterpretedContentBlock]] = {}
    section_stack: Dict[int, InterpretedDocumentSection] = {}

    for line in lines:
        candidate = heading_by_line_id.get(line.line_id)
        if candidate is not None:
            level = candidate["level"]
            parent = _nearest_parent(section_stack, level)
            section_id = f"pdf-section-{len(sections) + 1:04d}"
            heading_block = _content_block_from_line(
                line,
                section_id=section_id,
                role=CONTENT_ROLE_HEADING,
            )
            owned_blocks_by_section[section_id] = [heading_block]
            section = InterpretedDocumentSection(
                section_id=section_id,
                title=candidate["title"],
                logical_title=candidate["title"],
                level=level,
                parent_id=parent.section_id if parent else None,
                source_order=line.source_order,
                source_span=_span_from_line(line),
                confidence=candidate["confidence"],
                origin=candidate["origin"],
                owned_blocks=owned_blocks_by_section[section_id],
                metadata={
                    "line_id": line.line_id,
                    "source_block_ids": [block.block_id for block in line.blocks],
                    "detection_method": candidate["method"],
                    "numbering": candidate.get("numbering"),
                    "font_size": line.font_size,
                    "font_name": line.font_name,
                    "x": line.x,
                    "y": line.y,
                },
            )
            sections.append(section)
            active_section_id = section_id
            section_stack[level] = section
            for stale_level in list(section_stack):
                if stale_level > level:
                    del section_stack[stale_level]
            continue

        content_block = _content_block_from_line(
            line,
            section_id=active_section_id,
            role=CONTENT_ROLE_BODY,
        )
        if active_section_id:
            owned_blocks_by_section[active_section_id].append(content_block)
        else:
            unresolved_blocks.append(content_block)

    status = INTERPRETER_STATUS_PARTIAL if unresolved_blocks else INTERPRETER_STATUS_ACCEPTED
    result = DocumentInterpreterResult(
        document_id=document_id,
        source_format=SOURCE_FORMAT_PDF,
        document_title=document.title or document.filename,
        status=status,
        confidence=CONFIDENCE_MEDIUM if status == INTERPRETER_STATUS_PARTIAL else CONFIDENCE_HIGH,
        sections=sections,
        unresolved_blocks=unresolved_blocks,
        diagnostics={
            **diagnostics,
            "unresolved_blocks": len(unresolved_blocks),
        },
    )
    return _with_validation_diagnostics(result)


def _detect_heading_candidates(
    lines: List[PdfVisualLine],
    body_font_size: Optional[float],
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    explicit_root_line_ids = _explicit_root_numbering_line_ids(lines)
    for line in lines:
        numbered = _numbered_heading_candidate(
            line,
            body_font_size,
            force_root_line_id=line.line_id in explicit_root_line_ids,
        )
        if numbered:
            candidates.append(numbered)
            continue
        visual = _visual_heading_candidate(line, body_font_size, lines)
        if visual:
            candidates.append(visual)
    return candidates


def _explicit_root_numbering_line_ids(lines: List[PdfVisualLine]) -> Set[str]:
    raw_numbered = []
    for line in lines:
        text = " ".join((line.text or "").split()).strip()
        match = NUMBERED_HEADING_RE.match(text)
        if not match:
            continue
        numbering = match.group("number").strip()
        title = match.group("title").strip()
        delimiter = match.group("delimiter") or ""
        if _has_strong_numbered_rejection(numbering, delimiter, title, text):
            continue
        raw_numbered.append({
            "line": line,
            "numbering": numbering,
            "title": title,
            "delimiter": delimiter,
        })

    dotted = [item for item in raw_numbered if "." in item["numbering"]]
    roots = [item for item in raw_numbered if "." not in item["numbering"]]
    if not roots:
        return set()

    dotted_by_root: Dict[str, List[Dict[str, object]]] = {}
    for item in dotted:
        root = str(item["numbering"]).split(".", 1)[0]
        dotted_by_root.setdefault(root, []).append(item)

    selected: Dict[str, PdfVisualLine] = {}
    for root, children in dotted_by_root.items():
        first_child_order = min(child["line"].source_order for child in children)  # type: ignore[index]
        preceding_roots = [
            item
            for item in roots
            if item["numbering"] == root
            and item["line"].source_order < first_child_order  # type: ignore[index]
            and not _is_inside_dotted_subsection_span(item, dotted)
        ]
        if preceding_roots:
            selected[root] = max(
                preceding_roots,
                key=lambda item: item["line"].source_order,  # type: ignore[index]
            )["line"]  # type: ignore[assignment,index]

    dotted_roots = set(dotted_by_root)
    global_root_candidates = [
        item
        for item in roots
        if not _is_inside_dotted_subsection_span(item, dotted)
    ]
    global_numbers = sorted({
        int(str(item["numbering"]))
        for item in global_root_candidates
        if str(item["numbering"]).isdigit()
    })
    has_document_skeleton = (
        len(dotted_roots) >= 2
        and _has_coherent_integer_skeleton(global_numbers)
    )
    if has_document_skeleton:
        for number in global_numbers:
            key = str(number)
            if key in selected:
                continue
            matching = [
                item
                for item in global_root_candidates
                if item["numbering"] == key
            ]
            if matching:
                selected[key] = min(
                    matching,
                    key=lambda item: item["line"].source_order,  # type: ignore[index]
                )["line"]  # type: ignore[assignment,index]

    return {line.line_id for line in selected.values()}


def _has_strong_numbered_rejection(
    numbering: str,
    delimiter: str,
    title: str,
    text: str,
) -> bool:
    return (
        ")" in delimiter
        or "→" in title
        or _looks_like_prose(title)
        or _looks_like_page_or_year(numbering, title)
        or _looks_like_legal_reference(numbering, title)
        or _looks_like_duration_or_date(text)
    )


def _has_coherent_integer_skeleton(numbers: List[int]) -> bool:
    if len(numbers) < 4:
        return False
    longest_run = 1
    current_run = 1
    for left, right in zip(numbers, numbers[1:]):
        if right == left + 1:
            current_run += 1
        elif right != left:
            current_run = 1
        longest_run = max(longest_run, current_run)
    return longest_run >= 4


def _is_inside_dotted_subsection_span(
    root_item: Dict[str, object],
    dotted_items: List[Dict[str, object]],
) -> bool:
    line = root_item["line"]  # type: ignore[index]
    root_number = str(root_item["numbering"])
    before = [
        item
        for item in dotted_items
        if item["line"].source_order < line.source_order  # type: ignore[index,union-attr]
    ]
    after = [
        item
        for item in dotted_items
        if item["line"].source_order > line.source_order  # type: ignore[index,union-attr]
    ]
    if not before or not after:
        return False
    previous = max(before, key=lambda item: item["line"].source_order)  # type: ignore[index]
    following = min(after, key=lambda item: item["line"].source_order)  # type: ignore[index]
    previous_number = str(previous["numbering"])
    following_number = str(following["numbering"])
    previous_parent = previous_number.rsplit(".", 1)[0]
    following_parent = following_number.rsplit(".", 1)[0]
    if previous_parent != following_parent:
        return False
    return root_number != previous_parent


def _classify_pdf_mode(
    document: CanonicalDocumentInput,
    lines: List[PdfVisualLine],
) -> str:
    page_count = len(document.units)
    if page_count < 6:
        return "structured_notes"
    line_counts = [
        sum(1 for line in lines if line.page == unit.unit_index)
        for unit in document.units
    ]
    non_empty_pages = [count for count in line_counts if count > 0]
    if not non_empty_pages:
        return "structured_notes"
    median_lines = float(median(sorted(non_empty_pages)))
    title_like_pages = sum(
        1
        for unit in document.units
        if _slide_title_line_for_page(lines, unit.unit_index) is not None
    )
    title_ratio = title_like_pages / max(1, len(non_empty_pages))
    if median_lines <= 14 and title_ratio >= 0.45:
        return "slide_export"
    return "structured_notes"


def _interpret_slide_export_pdf(
    *,
    document: CanonicalDocumentInput,
    document_id: str,
    lines: List[PdfVisualLine],
    body_font_size: Optional[float],
) -> DocumentInterpreterResult:
    sections: List[InterpretedDocumentSection] = []
    diagnostics = {
        "page_count": len(document.units),
        "content_blocks": sum(len(line.blocks) for line in lines),
        "visual_lines": len(lines),
        "heading_count": 0,
        "numbered_heading_count": 0,
        "visual_heading_count": 0,
        "body_font_size": body_font_size,
        "pdf_mode": "slide_export",
        "titleless_slide_count": 0,
    }
    repeated_titles = _repeated_slide_title_texts(lines, document.units)

    for unit in document.units:
        page_lines = [line for line in lines if line.page == unit.unit_index]
        if not page_lines:
            continue
        title_line = _slide_title_line_for_page(
            lines,
            unit.unit_index,
            repeated_titles=repeated_titles,
        )
        title = title_line.text if title_line else f"Page {unit.unit_index}"
        confidence = CONFIDENCE_MEDIUM if title_line else CONFIDENCE_LOW
        section_id = f"pdf-slide-{unit.unit_index:04d}"
        owned_blocks = [
            _content_block_from_line(
                line,
                section_id=section_id,
                role=CONTENT_ROLE_HEADING if title_line and line.line_id == title_line.line_id else CONTENT_ROLE_BODY,
            )
            for line in page_lines
        ]
        sections.append(
            InterpretedDocumentSection(
                section_id=section_id,
                title=title,
                logical_title=title,
                level=1,
                parent_id=None,
                source_order=page_lines[0].source_order,
                source_span=SourceSpan(
                    start=page_lines[0].blocks[0].source_position,
                    end=page_lines[-1].blocks[-1].source_position,
                ),
                confidence=confidence,
                origin=ORIGIN_DETERMINISTIC_REPAIR,
                owned_blocks=owned_blocks,
                metadata={
                    "pdf_mode": "slide_export",
                    "page": unit.unit_index,
                    "title_line_id": title_line.line_id if title_line else None,
                },
            )
        )
        if title_line:
            diagnostics["heading_count"] += 1
            diagnostics["visual_heading_count"] += 1
        else:
            diagnostics["titleless_slide_count"] += 1

    if not sections:
        result = DocumentInterpreterResult(
            document_id=document_id,
            source_format=SOURCE_FORMAT_PDF,
            document_title=document.title or document.filename,
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[
                _content_block_from_line(line, section_id=None, role=CONTENT_ROLE_BODY)
                for line in lines
            ],
            diagnostics={
                **diagnostics,
                "fallback_reason": "slide_export_pdf_contains_no_extractable_lines",
            },
        )
        return _with_validation_diagnostics(result)

    status = (
        INTERPRETER_STATUS_PARTIAL
        if diagnostics["titleless_slide_count"]
        else INTERPRETER_STATUS_ACCEPTED
    )
    result = DocumentInterpreterResult(
        document_id=document_id,
        source_format=SOURCE_FORMAT_PDF,
        document_title=document.title or document.filename,
        status=status,
        confidence=CONFIDENCE_MEDIUM,
        sections=sections,
        diagnostics=diagnostics,
    )
    return _with_validation_diagnostics(result)


def _numbered_heading_candidate(
    line: PdfVisualLine,
    body_font_size: Optional[float],
    *,
    force_root_line_id: bool = False,
) -> Optional[Dict[str, object]]:
    text = " ".join((line.text or "").split()).strip()
    match = NUMBERED_HEADING_RE.match(text)
    if not match:
        return None
    title = match.group("title").strip()
    numbering = match.group("number").strip()
    delimiter = match.group("delimiter") or ""
    if _has_strong_numbered_rejection(numbering, delimiter, title, text):
        return None
    if _looks_like_numeric_measurement_range(numbering, title):
        return None
    if _looks_like_local_integer_content_item(numbering, delimiter, title):
        return None
    if _looks_like_local_decimal_list_item(numbering, delimiter, title) and not force_root_line_id:
        return None
    if (
        not force_root_line_id
        and not _has_heading_visual_support(line, body_font_size)
        and numbering.count(".") == 0
    ):
        return None
    return {
        "line_id": line.line_id,
        "title": text,
        "numbering": numbering,
        "level": numbering.count(".") + 1,
        "method": "numbered_heading",
        "confidence": CONFIDENCE_HIGH,
        "origin": ORIGIN_EXPLICIT_SOURCE,
    }


def _visual_heading_candidate(
    line: PdfVisualLine,
    body_font_size: Optional[float],
    all_lines: List[PdfVisualLine],
) -> Optional[Dict[str, object]]:
    text = " ".join((line.text or "").split()).strip()
    if not _text_can_be_visual_heading(text):
        return None
    if not _is_plausible_heading_position(line, all_lines):
        return None
    if _looks_like_local_decimal_visual_item(text, line, all_lines):
        return None
    if _looks_like_floating_diagram_label(line, all_lines):
        return None
    font_size = line.font_size
    if font_size is None or body_font_size is None:
        return None
    font_name = str(line.font_name or "").lower()
    is_bold = "bold" in font_name
    font_delta = font_size - body_font_size
    is_uppercase_heading = _is_mostly_uppercase(text) and (
        is_bold or _has_following_body_on_same_page(line, all_lines)
    )
    if font_delta < 3 and not (is_bold and font_delta >= 1.5) and not is_uppercase_heading:
        return None
    return {
        "line_id": line.line_id,
        "title": text,
        "numbering": None,
        "level": 1,
        "method": "visual_heading",
        "confidence": CONFIDENCE_MEDIUM,
        "origin": ORIGIN_DETERMINISTIC_REPAIR,
    }


def _slide_title_line_for_page(
    lines: List[PdfVisualLine],
    page: int,
    *,
    repeated_titles: Optional[Set[str]] = None,
) -> Optional[PdfVisualLine]:
    page_lines = [line for line in lines if line.page == page]
    if not page_lines:
        return None
    ordered = sorted(
        page_lines,
        key=lambda line: (
            -_float_or_default(line.y, -1_000_000.0),
            _float_or_default(line.x, 1_000_000.0),
            line.source_order,
        ),
    )
    candidates = [
        line
        for line in ordered
        if _is_plausible_slide_title_text(line.text)
    ]
    if not candidates:
        return None
    repeated_titles = repeated_titles or set()
    non_repeated = [
        line
        for line in candidates
        if _normalized_title_key(line.text) not in repeated_titles
    ]
    if repeated_titles and not non_repeated:
        return None
    best = max(non_repeated or candidates, key=lambda line: _slide_title_score(line, ordered))
    if _slide_title_score(best, ordered) < 4.5:
        return None
    return best


def _repeated_slide_title_texts(
    lines: List[PdfVisualLine],
    units,
) -> Set[str]:
    counts: Dict[str, int] = {}
    for unit in units:
        page_lines = sorted(
            [line for line in lines if line.page == unit.unit_index],
            key=lambda line: (
                -_float_or_default(line.y, -1_000_000.0),
                _float_or_default(line.x, 1_000_000.0),
                line.source_order,
            ),
        )
        for line in page_lines[:4]:
            if not _is_plausible_slide_title_text(line.text):
                continue
            key = _normalized_title_key(line.text)
            if key:
                counts[key] = counts.get(key, 0) + 1
    page_count = max(1, len(units))
    return {
        key
        for key, count in counts.items()
        if count >= 3 and count / page_count >= 0.2
    }


def _is_plausible_slide_title_text(text: str) -> bool:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return False
    if _looks_like_slide_title_fragment(normalized):
        return False
    if _looks_like_slide_footer_or_date(normalized):
        return False
    if _looks_like_prose(normalized):
        return False
    if len(normalized.split()) > 12 or len(normalized) > 120:
        return False
    if _looks_like_formula_or_diagram_fragment(normalized):
        return False
    return True


def _slide_title_score(line: PdfVisualLine, ordered_page_lines: List[PdfVisualLine]) -> float:
    text = " ".join((line.text or "").split()).strip()
    score = 0.0
    if len(text) >= 18:
        score += 1.0
    if _is_mostly_uppercase(text):
        score += 3.0
    if len(text.split()) >= 3:
        score += 2.0
    if ":" in text:
        score += 2.5
    if _has_structural_marker(text):
        score += 1.0
    page_left = min(
        (
            candidate.x
            for candidate in ordered_page_lines
            if candidate.x is not None and candidate.x > 5
        ),
        default=None,
    )
    if page_left is not None and line.x is not None and line.x <= page_left + 80:
        score += 1.5
    if line.x is not None and 70 <= line.x <= 190 and len(text.split()) >= 2:
        score += 2.0
    if line.y is not None:
        score += min(1.0, max(0.0, line.y / 700.0))
    return score


def _normalized_title_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _looks_like_formula_or_diagram_fragment(text: str) -> bool:
    normalized = " ".join((text or "").strip().split())
    if normalized in {".", ",", ";", ":", "=", "+", "-", "x", "y", "X", "Y"}:
        return True
    if len(normalized.split()) <= 3 and re.search(r"\b[A-Z]{1,3}[A-Z0-9]*\s+[A-Z]{1,3}[A-Z0-9]*\b", normalized):
        return True
    long_alpha_words = [
        word
        for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", normalized)
        if len(word) >= 4
    ]
    if len(long_alpha_words) >= 2:
        return False
    if len(normalized) <= 3 and re.fullmatch(r"[A-Za-z0-9+\-=]+", normalized):
        return True
    letters = [char for char in normalized if char.isalpha()]
    digits_or_symbols = [char for char in normalized if char.isdigit() or char in "+-=()[]"]
    if letters and digits_or_symbols and len(normalized.split()) <= 3:
        return True
    if normalized.startswith("(") and normalized.endswith(")") and len(normalized.split()) <= 2:
        return True
    return False


def _looks_like_slide_title_fragment(text: str) -> bool:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return True
    if normalized[0] in ":;,.=+-–—)]}":
        return True
    if normalized.endswith(","):
        return True
    words = normalized.split()
    if len(words) <= 4 and words[0].lower() in {"il", "lo", "la", "le", "gli", "i", "un", "una", "the"}:
        return True
    if len(words) <= 5 and _chemical_token_ratio(normalized) >= 0.6:
        return True
    return False


def _chemical_token_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9+\-=]+", text)
    if not tokens:
        return 0.0
    chemical_like = 0
    chemical_tokens = {"H", "C", "N", "O", "P", "S", "F", "CL", "BR", "I", "OH", "CN", "NO2", "HOMO", "LUMO"}
    for token in tokens:
        upper_token = token.upper()
        if upper_token in chemical_tokens:
            chemical_like += 1
            continue
        if re.fullmatch(r"[A-Z]{1,3}[0-9A-Z]*", token):
            chemical_like += 1
            continue
        if re.search(r"[A-Z][a-z]?[0-9]", token):
            chemical_like += 1
            continue
    return chemical_like / len(tokens)


def _text_can_be_visual_heading(text: str) -> bool:
    if not text or len(text) > 120:
        return False
    if len(text.split()) > 12:
        return False
    if _looks_like_dangling_heading_continuation(text):
        return False
    if re.match(r"^\d{1,2}(?:\.\d+)?\s+\S", text) and not _is_mostly_uppercase(text):
        return False
    normalized = " ".join(text.strip().split())
    if re.search(r"\.\s*$", normalized) and not _is_mostly_uppercase(normalized):
        return False
    if len(normalized.split()) <= 2 and re.search(r"[a-z][A-Z]", normalized):
        return False
    if _looks_like_prose(text):
        return False
    if re.match(r"^\d+$", text):
        return False
    if _looks_like_duration_or_date(text):
        return False
    if _looks_like_legal_reference("", text):
        return False
    return True


def _looks_like_dangling_heading_continuation(text: str) -> bool:
    normalized = " ".join((text or "").strip().upper().split())
    if not normalized:
        return False
    first = normalized.split()[0]
    return first in {"AND", "OR", "OF", "DELLA", "DELLE", "DEGLI", "DEI", "DI"} and len(normalized.split()) <= 4


def _looks_like_prose(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith((".", "?", "!", ";", ":")) and len(stripped.split()) > 6:
        return True
    return len(stripped.split()) > 16


def _looks_like_page_or_year(numbering: str, title: str) -> bool:
    if "." in numbering:
        return False
    number = int(numbering)
    if number > 80:
        return True
    if 1900 <= number <= 2100:
        return True
    return not title or len(title) < 3


def _looks_like_legal_reference(numbering: str, title: str) -> bool:
    normalized = " ".join(f"{numbering} {title}".lower().split())
    legal_tokens = (
        "cpp",
        "c.p.p",
        "cost.",
        "costituzione",
        "cedu",
        "art.",
        "art ",
        "co.",
        "comma",
        "d.lgs",
        "dpr",
        "c.c.",
        "cp",
    )
    if any(token in normalized for token in legal_tokens):
        return True
    if re.search(r"\b\d{1,3}\s*(cpp|c\.p\.p\.|cost\.|cedu|cp|c\.c\.)\b", normalized):
        return True
    return False


def _looks_like_duration_or_date(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    if re.search(r"\b\d{1,2}\s+(mesi|mese|giorni|giorno|anni|anno)\b", normalized):
        return True
    if re.search(
        r"\b\d{1,2}\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\b",
        normalized,
    ):
        return True
    return False


def _looks_like_slide_footer_or_date(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", normalized):
        return True
    if re.search(r"\bhttps?://|www\.|@\w", normalized):
        return True
    if re.search(r"\b\d+\s*/\s*\d+\b", normalized):
        return True
    return False


def _looks_like_local_decimal_list_item(
    numbering: str,
    delimiter: str,
    title: str,
) -> bool:
    if "." not in delimiter:
        return False
    if numbering.count(".") > 0:
        return False
    if _is_mostly_uppercase(title):
        return False
    normalized_title = " ".join((title or "").split()).strip()
    if normalized_title.upper().startswith("CHAPTER"):
        return False
    return True


def _looks_like_local_integer_content_item(
    numbering: str,
    delimiter: str,
    title: str,
) -> bool:
    if numbering.count(".") > 0:
        return False
    if "." in delimiter:
        return False
    normalized_title = " ".join((title or "").split()).strip()
    if not normalized_title:
        return True
    first_alpha = next((char for char in normalized_title if char.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return True
    if normalized_title.endswith(":") and not _is_mostly_uppercase(normalized_title):
        return True
    return False


def _looks_like_numeric_measurement_range(numbering: str, title: str) -> bool:
    normalized_title = " ".join((title or "").lower().split())
    if numbering.startswith("0.") and normalized_title.startswith("to "):
        return True
    if re.match(r"^to\s+\d", normalized_title):
        return True
    if re.search(r"\b(μ\s*m|µ\s*m|kda|mg|kg|mm|cm|nm)\b", normalized_title):
        return True
    return False


def _looks_like_local_decimal_visual_item(
    text: str,
    line: PdfVisualLine,
    all_lines: List[PdfVisualLine],
) -> bool:
    match = re.match(r"^\d{1,2}\.\s+(?P<title>\S.{1,100})$", text)
    if not match:
        return False
    title = match.group("title").strip()
    if _is_mostly_uppercase(title) or title.upper().startswith("CHAPTER"):
        return False
    page_lines_above = [
        candidate
        for candidate in all_lines
        if candidate.page == line.page
        and _float_or_default(candidate.y, -1_000_000.0) > _float_or_default(line.y, -1_000_000.0)
    ]
    has_prior_visual_anchor = any(
        _is_mostly_uppercase(candidate.text)
        or (
            candidate.font_size is not None
            and line.font_size is not None
            and candidate.font_size >= line.font_size + 2
        )
        for candidate in page_lines_above
    )
    has_intervening_body = any(_looks_like_prose(candidate.text) for candidate in page_lines_above)
    return has_prior_visual_anchor and has_intervening_body


def _has_following_body_on_same_page(
    line: PdfVisualLine,
    all_lines: List[PdfVisualLine],
) -> bool:
    following = [
        candidate
        for candidate in all_lines
        if candidate.page == line.page
        and candidate.source_order > line.source_order
    ]
    nearby = sorted(following, key=lambda candidate: candidate.source_order)[:5]
    return any(_looks_like_prose(candidate.text) for candidate in nearby)


def _has_heading_visual_support(
    line: PdfVisualLine,
    body_font_size: Optional[float],
) -> bool:
    if line.font_size is None or body_font_size is None:
        return False
    font_name = str(line.font_name or "").lower()
    font_delta = line.font_size - body_font_size
    if font_delta >= 2:
        return True
    if "bold" in font_name and font_delta >= 0:
        return True
    if _is_mostly_uppercase(line.text) and font_delta >= 0:
        return True
    return False


def _is_mostly_uppercase(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters) >= 0.72


def _median_font_size(blocks: List[CanonicalBlock]) -> Optional[float]:
    sizes = sorted(size for size in (_font_size(block) for block in blocks) if size is not None)
    if not sizes:
        return None
    return float(median(sizes))


def _body_font_size_from_lines(lines: List[PdfVisualLine]) -> Optional[float]:
    sizes = sorted(line.font_size for line in lines if line.font_size is not None)
    if not sizes:
        return None
    if len(sizes) <= 2:
        return float(sizes[0])
    lower_middle_index = max(0, (len(sizes) // 2) - 1)
    return float(sizes[lower_middle_index])


def _font_size(block: CanonicalBlock) -> Optional[float]:
    value = block.metadata.get("font_size")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _visual_lines_from_blocks(blocks: List[CanonicalBlock]) -> List[PdfVisualLine]:
    lines: List[PdfVisualLine] = []
    by_page: Dict[int, List[CanonicalBlock]] = {}
    for block in blocks:
        page = block.source_position.unit_index
        by_page.setdefault(page, []).append(block)

    for page, page_blocks in sorted(by_page.items()):
        sorted_blocks = sorted(
            page_blocks,
            key=lambda block: (
                -_float_or_default(block.metadata.get("y"), -1_000_000.0),
                _float_or_default(block.metadata.get("x"), 1_000_000.0),
                block.source_order,
            ),
        )
        line_groups: List[List[CanonicalBlock]] = []
        current: List[CanonicalBlock] = []
        current_y: Optional[float] = None
        for block in sorted_blocks:
            y = _float_or_none(block.metadata.get("y"))
            font_size = _font_size(block) or 12.0
            tolerance = max(2.5, font_size * 0.35)
            if current and current_y is not None and y is not None and abs(y - current_y) > tolerance:
                line_groups.append(current)
                current = []
                current_y = None
            current.append(block)
            if y is not None:
                current_y = y if current_y is None else (current_y + y) / 2
        if current:
            line_groups.append(current)

        for line_index, group in enumerate(line_groups, start=1):
            ordered_group = sorted(
                group,
                key=lambda block: (
                    _float_or_default(block.metadata.get("x"), 1_000_000.0),
                    block.source_order,
                ),
            )
            text = _merge_line_text([block.text for block in ordered_group])
            if not text:
                continue
            lines.append(
                PdfVisualLine(
                    line_id=f"page-{page:04d}-line-{line_index:04d}",
                    text=text,
                    raw_text="\n".join(block.raw_text for block in ordered_group),
                    source_order=min(block.source_order for block in ordered_group),
                    blocks=ordered_group,
                    page=page,
                    x=_first_number(block.metadata.get("x") for block in ordered_group),
                    y=_first_number(block.metadata.get("y") for block in ordered_group),
                    font_size=_dominant_font_size(ordered_group),
                    font_name=_dominant_font_name(ordered_group),
                )
            )

    lines = _merge_split_heading_lines(sorted(lines, key=lambda line: line.source_order))
    return _merge_heading_continuations(lines)


def _merge_split_heading_lines(lines: List[PdfVisualLine]) -> List[PdfVisualLine]:
    merged: List[PdfVisualLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if (
            index + 1 < len(lines)
            and _is_short_uppercase_fragment(current.text)
            and _is_mostly_uppercase(lines[index + 1].text)
            and current.page == lines[index + 1].page
            and _same_font_family(current.font_name, lines[index + 1].font_name)
        ):
            following = lines[index + 1]
            combined_blocks = current.blocks + following.blocks
            merged.append(
                PdfVisualLine(
                    line_id=current.line_id,
                    text=_merge_line_text([current.text, following.text]),
                    raw_text="\n".join([current.raw_text, following.raw_text]),
                    source_order=current.source_order,
                    blocks=combined_blocks,
                    page=current.page,
                    x=current.x,
                    y=current.y if current.y is not None else following.y,
                    font_size=current.font_size or following.font_size,
                    font_name=current.font_name or following.font_name,
                )
            )
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _is_short_uppercase_fragment(text: str) -> bool:
    normalized = "".join(char for char in (text or "").strip() if char.isalpha())
    return 1 <= len(normalized) <= 3 and normalized.isupper()


def _same_font_family(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return True
    return left.lower() == right.lower()


def _merge_heading_continuations(lines: List[PdfVisualLine]) -> List[PdfVisualLine]:
    merged: List[PdfVisualLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if index + 1 < len(lines) and _should_merge_heading_continuation(current, lines[index + 1]):
            following = lines[index + 1]
            combined_blocks = current.blocks + following.blocks
            merged.append(
                PdfVisualLine(
                    line_id=current.line_id,
                    text=_merge_line_text([current.text, following.text]),
                    raw_text="\n".join([current.raw_text, following.raw_text]),
                    source_order=current.source_order,
                    blocks=combined_blocks,
                    page=current.page,
                    x=current.x,
                    y=current.y if current.y is not None else following.y,
                    font_size=current.font_size or following.font_size,
                    font_name=current.font_name or following.font_name,
                )
            )
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _should_merge_heading_continuation(
    current: PdfVisualLine,
    following: PdfVisualLine,
) -> bool:
    if current.page != following.page:
        return False
    if not _same_font_family(current.font_name, following.font_name):
        return False
    if not (_is_heading_like_visual_text(current.text) and _is_heading_like_visual_text(following.text)):
        return False
    if not _is_close_vertical_continuation(current, following):
        return False
    if not _is_near_same_x(current, following, tolerance=55):
        return False
    return _ends_like_heading_continuation(current.text) or _has_structural_marker(current.text)


def _is_heading_like_visual_text(text: str) -> bool:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return False
    return _is_mostly_uppercase(normalized) or _has_structural_marker(normalized)


def _is_close_vertical_continuation(
    current: PdfVisualLine,
    following: PdfVisualLine,
) -> bool:
    if current.y is None or following.y is None:
        return abs(following.source_order - current.source_order) <= 3
    return 0 < abs(current.y - following.y) <= 32


def _is_near_same_x(
    current: PdfVisualLine,
    following: PdfVisualLine,
    *,
    tolerance: float,
) -> bool:
    if current.x is None or following.x is None:
        return True
    return abs(current.x - following.x) <= tolerance


def _ends_like_heading_continuation(text: str) -> bool:
    normalized = " ".join((text or "").upper().split())
    continuation_words = (
        "CONTRO",
        "NON",
        "DELLA",
        "DELLE",
        "DEGLI",
        "DEI",
        "DI",
        "E",
        "O",
    )
    return any(normalized.endswith(f" {word}") or normalized == word for word in continuation_words)


def _is_plausible_heading_position(
    line: PdfVisualLine,
    all_lines: List[PdfVisualLine],
) -> bool:
    if _is_top_page_title(line):
        return True
    if _is_near_left_margin(line, all_lines):
        return True
    if _has_structural_marker(line.text) and _is_near_left_margin(line, all_lines, tolerance=85):
        return True
    return False


def _looks_like_floating_diagram_label(
    line: PdfVisualLine,
    all_lines: List[PdfVisualLine],
) -> bool:
    if line.page == 1 or _has_structural_marker(line.text) or _is_top_page_title(line):
        return False
    text = " ".join((line.text or "").split()).strip()
    words = text.split()
    if not _is_mostly_uppercase(text):
        return False
    if "MODELLO" in text.upper() and len(words) >= 4:
        return True
    left_margin = _page_left_margin(line, all_lines)
    if left_margin is None or line.x is None:
        return False
    if len(words) <= 2 and line.x > left_margin + 18:
        return True
    return False


def _page_left_margin(
    line: PdfVisualLine,
    all_lines: List[PdfVisualLine],
) -> Optional[float]:
    page_x_values = [
        candidate.x
        for candidate in all_lines
        if candidate.page == line.page and candidate.x is not None and candidate.x > 5
    ]
    if not page_x_values:
        return None
    return min(page_x_values)


def _is_top_page_title(line: PdfVisualLine) -> bool:
    if line.page != 1:
        return False
    page_height = _line_page_height(line)
    if line.y is None or page_height is None:
        return line.source_order <= 3 and _is_mostly_uppercase(line.text)
    return line.y >= page_height * 0.86 and _is_mostly_uppercase(line.text)


def _is_near_left_margin(
    line: PdfVisualLine,
    all_lines: List[PdfVisualLine],
    *,
    tolerance: float = 38,
) -> bool:
    if line.x is None:
        return False
    left_margin = _page_left_margin(line, all_lines)
    if left_margin is None:
        return False
    return line.x <= left_margin + tolerance


def _line_page_height(line: PdfVisualLine) -> Optional[float]:
    for block in line.blocks:
        height = _float_or_none(block.metadata.get("page_height"))
        if height is not None:
            return height
    return None


def _has_structural_marker(text: str) -> bool:
    normalized = " ".join((text or "").strip().split())
    return bool(
        re.match(r"^[vq§}]\s+\S", normalized, flags=re.IGNORECASE)
        or re.match(r"^[A-Z]\.\s*\S", normalized)
    )


def _merge_line_text(parts: List[str]) -> str:
    merged = ""
    for part in parts:
        text = " ".join((part or "").split()).strip()
        if not text:
            continue
        if not merged:
            merged = text
            continue
        if _should_join_without_space(merged, text):
            merged += text
        else:
            merged += " " + text
    return " ".join(merged.split()).strip()


def _should_join_without_space(left: str, right: str) -> bool:
    if len(left) <= 2 and left.isupper() and right[:1].isupper():
        return True
    if left.endswith(("’", "'", "l’", "d’")):
        return True
    if right in {".", ",", ";", ":", ")", "]"}:
        return True
    if left.endswith("("):
        return True
    return False


def _dominant_font_size(blocks: List[CanonicalBlock]) -> Optional[float]:
    sizes = [_font_size(block) for block in blocks]
    sizes = [size for size in sizes if size is not None]
    if not sizes:
        return None
    return max(set(sizes), key=lambda size: (sizes.count(size), size))


def _dominant_font_name(blocks: List[CanonicalBlock]) -> Optional[str]:
    names = [str(block.metadata.get("font_name") or "") for block in blocks]
    names = [name for name in names if name]
    if not names:
        return None
    return max(set(names), key=lambda name: (names.count(name), name))


def _first_number(values) -> Optional[float]:
    for value in values:
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _float_or_default(value, default: float) -> float:
    number = _float_or_none(value)
    return number if number is not None else default
    try:
        return float(value)
    except Exception:
        return None


def _nearest_parent(
    section_stack: Dict[int, InterpretedDocumentSection],
    level: int,
) -> Optional[InterpretedDocumentSection]:
    for candidate_level in range(level - 1, 0, -1):
        parent = section_stack.get(candidate_level)
        if parent is not None:
            return parent
    return None


def _content_block(
    block: CanonicalBlock,
    *,
    section_id: Optional[str],
    role: str,
) -> InterpretedContentBlock:
    return InterpretedContentBlock(
        block_id=block.block_id,
        text=block.text,
        raw_text=block.raw_text,
        source_order=block.source_order,
        source_span=_span_from_block(block),
        role=role,
        page=block.source_position.unit_index if block.source_position.unit_type == "page" else None,
        section_id=section_id,
        provenance=ORIGIN_EXPLICIT_SOURCE,
        confidence=CONFIDENCE_HIGH if section_id else CONFIDENCE_LOW,
        metadata={
            "style_hint": block.style_hint,
            "native_hierarchy_hint": block.native_hierarchy_hint,
            **block.metadata,
        },
    )


def _content_block_from_line(
    line: PdfVisualLine,
    *,
    section_id: Optional[str],
    role: str,
) -> InterpretedContentBlock:
    return InterpretedContentBlock(
        block_id=line.line_id,
        text=line.text,
        raw_text=line.raw_text,
        source_order=line.source_order,
        source_span=_span_from_line(line),
        role=role,
        page=line.page,
        section_id=section_id,
        provenance=ORIGIN_EXPLICIT_SOURCE,
        confidence=CONFIDENCE_HIGH if section_id else CONFIDENCE_LOW,
        metadata={
            "source_block_ids": [block.block_id for block in line.blocks],
            "x": line.x,
            "y": line.y,
            "font_size": line.font_size,
            "font_name": line.font_name,
        },
    )


def _span_from_block(block: CanonicalBlock) -> SourceSpan:
    return SourceSpan(
        start=block.source_position,
        end=block.source_position,
    )


def _span_from_line(line: PdfVisualLine) -> SourceSpan:
    return SourceSpan(
        start=line.blocks[0].source_position,
        end=line.blocks[-1].source_position,
    )


def _with_validation_diagnostics(
    result: DocumentInterpreterResult,
) -> DocumentInterpreterResult:
    report = validate_document_interpreter_result(result)
    return DocumentInterpreterResult(
        document_id=result.document_id,
        source_format=result.source_format,
        document_title=result.document_title,
        status=result.status,
        confidence=result.confidence,
        sections=result.sections,
        unresolved_blocks=result.unresolved_blocks,
        diagnostics={
            **result.diagnostics,
            "contract_validation": {
                "is_valid": report.is_valid,
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "object_id": issue.object_id,
                    }
                    for issue in report.issues
                ],
            },
        },
    )
