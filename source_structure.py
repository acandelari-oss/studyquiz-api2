import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


SOURCE_STRUCTURE_HIGH = "HIGH"
SOURCE_STRUCTURE_MEDIUM = "MEDIUM"
SOURCE_STRUCTURE_LOW = "LOW"

STATUS_ACCEPTED = "accepted"
STATUS_UNCERTAIN = "uncertain"
STATUS_REJECTED = "rejected"

SCOPE_GLOBAL = "global"
SCOPE_LOCAL = "local"
SCOPE_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class SourceStructureSignal:
    id: str
    signal_type: str
    source_page: Optional[int]
    source_block: int
    source_line: int
    strength: float
    explanation: str
    target_id: Optional[str] = None


@dataclass(frozen=True)
class SourceStructureNode:
    id: str
    title_original: str
    title_normalized: str
    numbering: Optional[str]
    hierarchy_level: int
    parent_id: Optional[str]
    source_page: Optional[int]
    source_block: int
    source_line: int
    source_order: int
    detection_method: str
    region_id: Optional[str]
    sequence_id: Optional[str]
    section_scope_id: Optional[str]
    structural_scope: str
    status: str
    confidence: str
    confidence_score: float
    supporting_signal_ids: List[str] = field(default_factory=list)
    negative_signal_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class SourceStructureRelationship:
    id: str
    parent_node_id: str
    child_node_id: str
    relationship_type: str
    confidence: str
    confidence_score: float
    supporting_signal_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class SourceStructureDocument:
    filename: str
    file_format: str
    file_size_bytes: int
    pages_detected: Optional[int]
    confidence: str
    accepted_node_count: int
    uncertain_node_count: int
    rejected_candidate_count: int


@dataclass(frozen=True)
class SourceStructureRegion:
    id: str
    region_type: str
    confidence: str
    start_page: Optional[int]
    end_page: Optional[int]
    start_block: int
    end_block: int
    start_line: int
    end_line: int
    candidate_node_ids: List[str] = field(default_factory=list)
    supporting_signal_ids: List[str] = field(default_factory=list)
    negative_signal_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class SourceStructureSequence:
    id: str
    sequence_type: str
    confidence: str
    candidate_node_ids: List[str] = field(default_factory=list)
    region_id: Optional[str] = None
    supporting_signal_ids: List[str] = field(default_factory=list)
    negative_signal_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class SourceStructureScope:
    id: str
    opener_node_id: str
    opener_title: str
    confidence: str
    start_page: Optional[int]
    end_page: Optional[int]
    start_block: int
    end_block: int
    start_line: int
    end_line: int
    contained_region_ids: List[str] = field(default_factory=list)
    contained_sequence_ids: List[str] = field(default_factory=list)
    supporting_signal_ids: List[str] = field(default_factory=list)
    negative_signal_ids: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class SourceHeading:
    """Backward-compatible diagnostic heading used by existing upload logs."""

    text: str
    normalized_title: str
    numbering: Optional[str]
    hierarchy_level: int
    source_page: Optional[int]
    source_block: int
    source_line: int
    document_order: int
    detection_method: str
    region_id: Optional[str] = None
    sequence_id: Optional[str] = None
    section_scope_id: Optional[str] = None
    structural_scope: str = SCOPE_UNRESOLVED
    status: str = STATUS_ACCEPTED
    confidence: str = SOURCE_STRUCTURE_HIGH
    confidence_score: float = 1.0
    parent_id: Optional[str] = None
    node_id: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class SourceStructureAnalysis:
    source_structure_confidence: str
    global_structure_confidence: str
    local_structure_confidence: str
    headings: List[SourceHeading]
    numbered_heading_count: int
    textual_heading_count: int
    hierarchy_depth: int
    strongest_parent: Optional[str]
    strongest_parent_child_count: int
    document: Optional[SourceStructureDocument] = None
    nodes: List[SourceStructureNode] = field(default_factory=list)
    relationships: List[SourceStructureRelationship] = field(default_factory=list)
    signals: List[SourceStructureSignal] = field(default_factory=list)
    regions: List[SourceStructureRegion] = field(default_factory=list)
    sequences: List[SourceStructureSequence] = field(default_factory=list)
    section_scopes: List[SourceStructureScope] = field(default_factory=list)
    accepted_node_count: int = 0
    uncertain_node_count: int = 0
    rejected_candidate_count: int = 0


@dataclass
class _Candidate:
    id: str
    raw_text: str
    title: str
    normalized_title: str
    numbering: Optional[str]
    hierarchy_level: int
    source_page: Optional[int]
    source_block: int
    source_line: int
    source_order: int
    detection_method: str
    region_id: Optional[str] = None
    sequence_id: Optional[str] = None
    section_scope_id: Optional[str] = None
    original_lines: List[str] = field(default_factory=list)
    structural_scope: str = SCOPE_UNRESOLVED
    status: str = STATUS_UNCERTAIN
    confidence: str = SOURCE_STRUCTURE_LOW
    confidence_score: float = 0.0
    parent_id: Optional[str] = None
    supporting_signal_ids: List[str] = field(default_factory=list)
    negative_signal_ids: List[str] = field(default_factory=list)
    reason: str = ""
    consumed_lines: int = 1


@dataclass(frozen=True)
class _SkeletonSupport:
    confidence: str
    score: float
    reason: str


@dataclass(frozen=True)
class _GlobalNumberingSkeleton:
    supported: Dict[str, _SkeletonSupport] = field(default_factory=dict)
    excluded: Dict[str, str] = field(default_factory=dict)
    contained_local_integer_ids: set = field(default_factory=set)


_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2}){0,4})\s+"
    r"(?P<title>[^\d\s][^\n]{2,140})\s*$"
)
_ACADEMIC_DOTTED_TOP_LEVEL_RE = re.compile(
    r"^\s*(?P<number>\d{1,2})\.\s+"
    r"(?P<title>[^\d\s][^\n]{2,140})\s*$"
)
_PROCEDURAL_DOTTED_SENTENCE_RE = re.compile(
    r"^\s*(?P<number>\d{1,4})\.\s+"
    r"(?P<sentence>(?:the|a|an|no|if|when|then|this|these|it|la|il|lo|gli|le|un|una|no)\b.+)",
    re.IGNORECASE,
)
_STANDALONE_SECTION_NUMBER_RE = re.compile(r"^\s*(?P<number>\d{1,2})\s*$")
_ROMAN_RE = re.compile(r"^\s*(?P<roman>[IVXLCDM]{1,8})\.?\s+(?P<title>[^\n]{2,120})\s*$")
_ROMAN_STANDALONE_RE = re.compile(r"^\s*(?P<roman>[IVXLCDM]{1,8})\.?\s*$")
_STRUCTURAL_LABEL_RE = re.compile(
    r"^\s*(?P<label>PART|PARTE|CHAPTER|CAPITOLO|SECTION|SEZIONE)\s+"
    r"(?P<rest>[^\n]{2,140})\s*$",
    re.IGNORECASE,
)
_TEXTUAL_HEADING_MAX_WORDS = 9
_MAX_TOP_LEVEL_SECTION = 30
_MAX_PROCEDURAL_NUMBER = 99
_YEAR_RE = re.compile(r"^(?:18|19|20)\d{2}$")
_MOSTLY_PUNCTUATION_RE = re.compile(r"^[\W\d_]+$")
_FORMULA_MARKERS_RE = re.compile(r"→|←|↔")
_CITATION_LINE_RE = re.compile(
    r"\b(?:doi|isbn|issn|vol\.|pp\.|pages?|et al\.|journal|"
    r"proceedings|conference|references|bibliography)\b",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^(?:page|pagina|pag\.?)?\s*\d{1,4}\s*$", re.IGNORECASE)
_SENTENCE_START_RE = re.compile(
    r"^(?:the|a|an|no|if|when|then|this|these|it|there|la|il|lo|gli|le|un|una)\b",
    re.IGNORECASE,
)
_LOWERCASE_START_RE = re.compile(r"^[a-zà-ÿ]")
_INTERROGATIVE_START_RE = re.compile(
    r"^(?:cosa|come|quante|quanti|quali|qual|perché|perche|quando|dove|chi|"
    r"what|how|which|why|when|where|who)\b",
    re.IGNORECASE,
)
_PROSE_CONTINUATION_START_RE = re.compile(
    r"^(?:si|negli|nelle|nel|nella|nei|esso|essa|può|puo|anche|"
    r"inoltre|tuttavia|therefore|however|moreover|also)\b",
    re.IGNORECASE,
)

_ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}
_LABEL_LEVELS = {
    "part": 1,
    "parte": 1,
    "chapter": 2,
    "capitolo": 2,
    "section": 3,
    "sezione": 3,
}


def analyze_source_structure(extracted_document) -> SourceStructureAnalysis:
    """Build a diagnostic source-structure model without affecting taxonomy.

    The analysis is intentionally evidence based:
    candidate extraction is permissive, validation is conservative, hierarchy
    construction is explicit, and confidence is derived from named signals.
    """

    candidates, signals = _extract_candidates(extracted_document)
    _reconstruct_multiline_headings(candidates, signals)
    regions = _segment_regions(candidates, signals)
    sequences = _classify_numbered_sequences(candidates, regions, signals)
    global_skeleton = _compute_global_numbering_skeleton(candidates, regions, sequences, signals)
    _validate_candidates(candidates, signals, regions, sequences, global_skeleton)
    _rehabilitate_explicit_numbering_parents(candidates, signals, regions, sequences, global_skeleton)
    section_scopes = _build_section_scopes(candidates, regions, sequences, signals, global_skeleton)
    _assign_structural_scopes(candidates, regions, sequences, section_scopes, global_skeleton)
    relationships = _construct_hierarchy(candidates, signals, regions, section_scopes, global_skeleton)
    global_confidence, local_confidence = _evaluate_structure_confidence(
        candidates,
        relationships,
        regions,
        sequences,
        section_scopes,
    )

    accepted_candidates = [c for c in candidates if c.status == STATUS_ACCEPTED]
    uncertain_candidates = [c for c in candidates if c.status == STATUS_UNCERTAIN]
    rejected_candidates = [c for c in candidates if c.status == STATUS_REJECTED]

    nodes = [
        SourceStructureNode(
            id=c.id,
            title_original=_display_title(c),
            title_normalized=c.normalized_title,
            numbering=c.numbering,
            hierarchy_level=c.hierarchy_level,
            parent_id=c.parent_id,
            source_page=c.source_page,
            source_block=c.source_block,
            source_line=c.source_line,
            source_order=c.source_order,
            detection_method=c.detection_method,
            region_id=c.region_id,
            sequence_id=c.sequence_id,
            section_scope_id=c.section_scope_id,
            structural_scope=c.structural_scope,
            status=c.status,
            confidence=c.confidence,
            confidence_score=round(c.confidence_score, 3),
            supporting_signal_ids=list(c.supporting_signal_ids),
            negative_signal_ids=list(c.negative_signal_ids),
            reason=c.reason,
        )
        for c in candidates
    ]

    visible_headings = [
        _candidate_to_heading(c)
        for c in candidates
        if c.status in {STATUS_ACCEPTED, STATUS_UNCERTAIN}
    ]

    accepted_numbered = [
        c for c in accepted_candidates
        if c.detection_method in {"numbered", "split_numbered", "academic_numbered"}
    ]
    hierarchy_stats = _numbered_hierarchy_stats(
        _candidate_to_heading(c) for c in accepted_numbered
    )
    textual_count = len([
        c for c in candidates
        if c.status in {STATUS_ACCEPTED, STATUS_UNCERTAIN}
        and c.detection_method == "textual"
    ])

    document = SourceStructureDocument(
        filename=getattr(extracted_document, "filename", ""),
        file_format=getattr(extracted_document, "file_format", ""),
        file_size_bytes=getattr(extracted_document, "file_size_bytes", 0),
        pages_detected=getattr(extracted_document, "pages_detected", None),
        confidence=global_confidence,
        accepted_node_count=len(accepted_candidates),
        uncertain_node_count=len(uncertain_candidates),
        rejected_candidate_count=len(rejected_candidates),
    )

    return SourceStructureAnalysis(
        source_structure_confidence=global_confidence,
        global_structure_confidence=global_confidence,
        local_structure_confidence=local_confidence,
        headings=visible_headings,
        numbered_heading_count=len(accepted_numbered),
        textual_heading_count=textual_count,
        hierarchy_depth=_max_hierarchy_depth(accepted_candidates),
        strongest_parent=hierarchy_stats["strongest_parent"],
        strongest_parent_child_count=hierarchy_stats["strongest_parent_child_count"],
        document=document,
        nodes=nodes,
        relationships=relationships,
        signals=signals,
        regions=regions,
        sequences=sequences,
        section_scopes=section_scopes,
        accepted_node_count=len(accepted_candidates),
        uncertain_node_count=len(uncertain_candidates),
        rejected_candidate_count=len(rejected_candidates),
    )


def format_source_structure_tree(analysis: SourceStructureAnalysis) -> str:
    lines = [
        f"source_structure_confidence: {analysis.source_structure_confidence}",
        f"global_structure_confidence: {analysis.global_structure_confidence}",
        f"local_structure_confidence: {analysis.local_structure_confidence}",
        f"document_confidence: {analysis.document.confidence if analysis.document else analysis.source_structure_confidence}",
        (
            "nodes: "
            f"accepted={analysis.accepted_node_count}, "
            f"uncertain={analysis.uncertain_node_count}, "
            f"rejected={analysis.rejected_candidate_count}"
        ),
        "",
        "TREE",
    ]

    accepted_nodes = [
        node for node in analysis.nodes
        if node.status == STATUS_ACCEPTED
    ]
    relationship_by_child = {
        relationship.child_node_id: relationship
        for relationship in analysis.relationships
    }
    node_by_id = {node.id: node for node in analysis.nodes}

    def display_depth(node: SourceStructureNode) -> int:
        depth = 0
        seen = set()
        parent_id = node.parent_id
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = node_by_id.get(parent_id)
            if not parent:
                break
            depth += 1
            parent_id = parent.parent_id
        return depth

    if not accepted_nodes:
        lines.append("(no accepted structural nodes)")
    else:
        for node in accepted_nodes:
            relationship = relationship_by_child.get(node.id)
            indent = "  " * display_depth(node)
            location = (
                f"method={node.detection_method}"
                f", numbering={node.numbering or 'n/a'}"
                f", level={node.hierarchy_level}"
                f", scope={node.structural_scope}"
                f", confidence={node.confidence}"
                f", region={node.region_id or 'n/a'}"
                f", sequence={node.sequence_id or 'n/a'}"
                f", section_scope={node.section_scope_id or 'n/a'}"
                f", relationship={relationship.relationship_type if relationship else 'n/a'}"
                f", page={node.source_page if node.source_page is not None else 'n/a'}"
                f", block={node.source_block}"
                f", line={node.source_line}"
                f", order={node.source_order}"
            )
            lines.append(f"{indent}{node.title_original} [{location}]")

    lines.extend(["", "SECTION SCOPES"])
    if not analysis.section_scopes:
        lines.append("(no section scopes detected)")
    else:
        for section_scope in analysis.section_scopes:
            lines.append(
                f"- {section_scope.id}: opener={section_scope.opener_title!r}, "
                f"confidence={section_scope.confidence}, "
                f"range=page {section_scope.start_page if section_scope.start_page is not None else 'n/a'}"
                f"-{section_scope.end_page if section_scope.end_page is not None else 'n/a'}, "
                f"block {section_scope.start_block}-{section_scope.end_block}, "
                f"line {section_scope.start_line}-{section_scope.end_line}; "
                f"regions={', '.join(section_scope.contained_region_ids) or 'n/a'}; "
                f"sequences={', '.join(section_scope.contained_sequence_ids) or 'n/a'}; "
                f"reason={section_scope.reason}"
            )

    lines.extend(["", "REGIONS"])
    if not analysis.regions:
        lines.append("(no source regions detected)")
    else:
        for region in analysis.regions:
            lines.append(
                f"- {region.id}: type={region.region_type}, confidence={region.confidence}, "
                f"range=page {region.start_page if region.start_page is not None else 'n/a'}"
                f"-{region.end_page if region.end_page is not None else 'n/a'}, "
                f"block {region.start_block}-{region.end_block}, "
                f"line {region.start_line}-{region.end_line}; "
                f"candidates={len(region.candidate_node_ids)}; reason={region.reason}"
            )

    lines.extend(["", "SEQUENCES"])
    if not analysis.sequences:
        lines.append("(no numbered sequences detected)")
    else:
        for sequence in analysis.sequences:
            lines.append(
                f"- {sequence.id}: type={sequence.sequence_type}, confidence={sequence.confidence}, "
                f"region={sequence.region_id or 'n/a'}, candidates={len(sequence.candidate_node_ids)}, "
                f"reason={sequence.reason}"
            )

    lines.extend(["", "RELATIONSHIPS"])
    if not analysis.relationships:
        lines.append("(no accepted parent-child relationships)")
    else:
        by_id = {node.id: node for node in analysis.nodes}
        for relationship in analysis.relationships:
            parent = by_id.get(relationship.parent_node_id)
            child = by_id.get(relationship.child_node_id)
            lines.append(
                f"- {parent.title_original if parent else relationship.parent_node_id}"
                f" -> {child.title_original if child else relationship.child_node_id}"
                f" [{relationship.relationship_type}, confidence={relationship.confidence}, "
                f"reason={relationship.reason}]"
            )

    rejected_or_uncertain = [
        node for node in analysis.nodes
        if node.status in {STATUS_REJECTED, STATUS_UNCERTAIN}
    ]
    lines.extend(["", "REJECTED / UNCERTAIN CANDIDATES"])
    if not rejected_or_uncertain:
        lines.append("(none)")
    else:
        for node in rejected_or_uncertain:
            lines.append(
                f"- {node.status.upper()}: {node.title_original!r} "
                f"[method={node.detection_method}, page={node.source_page if node.source_page is not None else 'n/a'}, "
                f"block={node.source_block}, line={node.source_line}, region={node.region_id or 'n/a'}, "
                f"reason={node.reason}]"
            )

    lines.extend(["", "SIGNALS"])
    important_signals = [
        signal for signal in analysis.signals
        if abs(signal.strength) >= 0.45
    ]
    if not important_signals:
        lines.append("(no strong diagnostic signals)")
    else:
        for signal in important_signals[:80]:
            lines.append(
                f"- {signal.id}: {signal.signal_type} "
                f"(strength={signal.strength:+.2f}, target={signal.target_id or 'document'}, "
                f"page={signal.source_page if signal.source_page is not None else 'n/a'}, "
                f"block={signal.source_block}, line={signal.source_line}) — {signal.explanation}"
            )

    return "\n".join(lines)


def source_structure_to_dict(analysis: SourceStructureAnalysis) -> dict:
    return {
        "source_structure_confidence": analysis.source_structure_confidence,
        "global_structure_confidence": analysis.global_structure_confidence,
        "local_structure_confidence": analysis.local_structure_confidence,
        "document": asdict(analysis.document) if analysis.document else None,
        "numbered_heading_count": analysis.numbered_heading_count,
        "textual_heading_count": analysis.textual_heading_count,
        "hierarchy_depth": analysis.hierarchy_depth,
        "strongest_parent": analysis.strongest_parent,
        "strongest_parent_child_count": analysis.strongest_parent_child_count,
        "accepted_node_count": analysis.accepted_node_count,
        "uncertain_node_count": analysis.uncertain_node_count,
        "rejected_candidate_count": analysis.rejected_candidate_count,
        "headings": [asdict(heading) for heading in analysis.headings],
        "nodes": [asdict(node) for node in analysis.nodes],
        "relationships": [asdict(relationship) for relationship in analysis.relationships],
        "signals": [asdict(signal) for signal in analysis.signals],
        "regions": [asdict(region) for region in analysis.regions],
        "sequences": [asdict(sequence) for sequence in analysis.sequences],
        "section_scopes": [asdict(section_scope) for section_scope in analysis.section_scopes],
    }


def source_structure_to_json(analysis: SourceStructureAnalysis) -> str:
    return json.dumps(source_structure_to_dict(analysis), ensure_ascii=False, indent=2)


def _extract_candidates(extracted_document) -> Tuple[List[_Candidate], List[SourceStructureSignal]]:
    candidates: List[_Candidate] = []
    signals: List[SourceStructureSignal] = []
    order = 0

    for block_index, block in enumerate(getattr(extracted_document, "blocks", []) or [], start=1):
        raw_text = getattr(block, "raw_text", "") or getattr(block, "text", "") or ""
        page = getattr(block, "page", None)
        source_block = getattr(block, "block_index", None) or block_index
        lines = raw_text.splitlines()

        line_index = 1
        while line_index <= len(lines):
            line = lines[line_index - 1]
            clean_line = _normalize_line(line)
            if not clean_line:
                line_index += 1
                continue

            candidate = (
                _extract_procedural_numbered_sentence(clean_line, page, source_block, line_index, order)
                or _extract_academic_dotted_heading(clean_line, page, source_block, line_index, order)
                or _extract_numbered_heading(clean_line, page, source_block, line_index, order)
                or _extract_structural_label(clean_line, lines, line_index, page, source_block, order)
                or _extract_roman_heading(clean_line, page, source_block, line_index, order)
                or _extract_split_numbered_heading(clean_line, lines, line_index, page, source_block, order)
                or _extract_split_roman_heading(clean_line, lines, line_index, page, source_block, order)
                or _extract_textual_heading(clean_line, lines, line_index, page, source_block, order)
            )

            if candidate:
                candidates.append(candidate)
                _add_signal(
                    signals,
                    candidate,
                    _initial_signal_for_candidate(candidate),
                    _initial_strength_for_candidate(candidate),
                    f"Candidate extracted by {candidate.detection_method}",
                )
                order += 1

                if candidate.consumed_lines > 1:
                    next_line_index, _ = _next_non_empty_line(lines, line_index)
                    line_index = (next_line_index or line_index) + candidate.consumed_lines - 1
                    continue

                if candidate.detection_method in {"split_numbered", "split_roman"}:
                    next_line_index, _ = _next_non_empty_line(lines, line_index)
                    line_index = (next_line_index or line_index) + 1
                    continue

            line_index += 1

    return candidates, signals


def _validate_candidates(
    candidates: List[_Candidate],
    signals: List[SourceStructureSignal],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    global_skeleton: _GlobalNumberingSkeleton,
) -> None:
    contained_ids = global_skeleton.contained_local_integer_ids
    numbered_groups = _numbered_child_groups(candidates, contained_ids)
    top_level_numbers = _ordered_top_level_numbers(candidates, contained_ids)
    roman_sequence = _ordered_roman_numbers(candidates)
    textual_patterns = _textual_heading_patterns(candidates)
    structural_label_context = _has_structural_label_context(candidates)

    for candidate in candidates:
        if candidate.detection_method == "procedural_numbered_sentence":
            _reject(candidate, signals, "procedural_numbering_penalty", "Procedural numbered sentence, not a structural heading")
            continue

        strong_issue = _strong_heading_title_safety_issue(candidate.title)
        weak_issue = _weak_heading_title_safety_issue(candidate.title)
        sequence = _sequence_for_candidate(candidate, sequences)
        sequence_can_rehabilitate = (
            sequence is not None
            and sequence.sequence_type in {
                "semantic_local_sequence",
                "structural_sequence",
                "toc_sequence",
            }
        )
        skeleton_can_rehabilitate = candidate.id in global_skeleton.supported
        if strong_issue or (weak_issue and not sequence_can_rehabilitate and not skeleton_can_rehabilitate):
            _reject(candidate, signals, "unsafe_title_penalty", "Candidate title matches page/reference/formula/sentence false-positive protection")
            continue
        if weak_issue and sequence_can_rehabilitate:
            _add_signal(
                signals,
                candidate,
                "sequence_rehabilitated_weak_title",
                0.42,
                "Weak title-shape issue was overridden by strong contiguous sequence evidence",
            )

        if candidate.detection_method in {"numbered", "academic_numbered", "split_numbered"}:
            _validate_numbered_candidate(candidate, candidates, signals, numbered_groups, top_level_numbers, regions, sequences, global_skeleton)
            continue

        if candidate.detection_method in {"roman", "split_roman"}:
            _validate_roman_candidate(candidate, signals, roman_sequence)
            continue

        if candidate.detection_method == "structural_label":
            _accept(candidate, signals, SOURCE_STRUCTURE_HIGH, 0.9, "explicit_structural_label", "Explicit PART/CHAPTER/SECTION style label")
            continue

        if candidate.detection_method == "textual":
            _validate_textual_candidate(candidate, signals, textual_patterns, structural_label_context, regions)
            continue

        _uncertain(candidate, signals, "insufficient_evidence", "Candidate has no strong structural context")


def _reconstruct_multiline_headings(
    candidates: List[_Candidate],
    signals: List[SourceStructureSignal],
) -> None:
    for candidate in candidates:
        if candidate.consumed_lines > 1:
            _add_signal(
                signals,
                candidate,
                "multiline_heading_continuation_support",
                0.78,
                "Strong structural heading merged with immediately adjacent heading-like continuation line",
            )


def _segment_regions(
    candidates: List[_Candidate],
    signals: List[SourceStructureSignal],
) -> List[SourceStructureRegion]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: (c.source_block, c.source_line, c.source_order))
    raw_regions: List[List[_Candidate]] = []
    current: List[_Candidate] = []

    for candidate in sorted_candidates:
        if not current:
            current = [candidate]
            continue

        previous = current[-1]
        same_block = previous.source_block == candidate.source_block
        line_gap = candidate.source_line - previous.source_line if same_block else 999
        boundary = (
            not same_block
            or line_gap > 8
            or (
                candidate.detection_method == "structural_label"
                and current
                and len(current) >= 2
            )
            or (
                candidate.detection_method == "textual"
                and any(existing.numbering for existing in current)
            )
        )
        if boundary:
            raw_regions.append(current)
            current = [candidate]
        else:
            current.append(candidate)

    if current:
        raw_regions.append(current)

    regions: List[SourceStructureRegion] = []
    for index, region_candidates in enumerate(raw_regions, start=1):
        region_id = f"region-{index:04d}"
        region_type, confidence, reason, positive_type = _classify_region(region_candidates)
        signal_ids = []
        negative_ids = []
        region_signal = _add_region_signal(
            signals,
            region_id,
            region_candidates[0],
            positive_type,
            _region_signal_strength(confidence, region_type),
            reason,
        )
        if region_signal.strength > 0:
            signal_ids.append(region_signal.id)
        else:
            negative_ids.append(region_signal.id)

        for candidate in region_candidates:
            candidate.region_id = region_id
        if region_type in {"toc_like", "structural_outline", "semantic_local_scope"}:
            _apply_region_hierarchy_context(region_candidates, region_type)

        pages = [
            c.source_page for c in region_candidates
            if c.source_page is not None
        ]
        regions.append(
            SourceStructureRegion(
                id=region_id,
                region_type=region_type,
                confidence=confidence,
                start_page=min(pages) if pages else None,
                end_page=max(pages) if pages else None,
                start_block=region_candidates[0].source_block,
                end_block=region_candidates[-1].source_block,
                start_line=region_candidates[0].source_line,
                end_line=region_candidates[-1].source_line,
                candidate_node_ids=[c.id for c in region_candidates],
                supporting_signal_ids=signal_ids,
                negative_signal_ids=negative_ids,
                reason=reason,
            )
        )

    return regions


def _classify_region(candidates: List[_Candidate]) -> Tuple[str, str, str, str]:
    count = len(candidates)
    structural_labels = [c for c in candidates if c.detection_method == "structural_label"]
    numbered = [c for c in candidates if c.numbering and c.numbering.isdigit()]
    nested = [c for c in candidates if c.numbering and "." in c.numbering]
    textual = [c for c in candidates if c.detection_method == "textual"]
    procedural = [c for c in candidates if c.detection_method == "procedural_numbered_sentence"]
    dense = _is_dense_region(candidates)

    if procedural and len(procedural) >= max(2, count // 2):
        return (
            "procedural",
            SOURCE_STRUCTURE_HIGH,
            "Region contains sentence-like sequential procedural numbering",
            "procedural_sequence_support",
        )

    if structural_labels and (nested or numbered or len(structural_labels) >= 2):
        return (
            "toc_like" if dense else "structural_outline",
            SOURCE_STRUCTURE_HIGH,
            "Region contains explicit structural labels with compact hierarchy/numbering evidence",
            "toc_density_support" if dense else "structural_region_support",
        )

    if (
        textual
        and numbered
        and len(numbered) >= 3
        and not nested
        and textual[0].source_order < numbered[0].source_order
        and numbered[0].source_line - textual[-1].source_line <= 4
        and not _numbered_run_is_predominantly_content_like(numbered)
        and not _looks_like_procedural_region(candidates)
    ):
        return (
            "semantic_local_scope",
            SOURCE_STRUCTURE_HIGH,
            "Textual heading is followed by a coherent local numbered structure",
            "semantic_local_scope_support",
        )

    if nested and len(nested) >= 2:
        return (
            "structural_outline",
            SOURCE_STRUCTURE_HIGH,
            "Region contains nested numbered hierarchy candidates",
            "structural_region_support",
        )

    if numbered and len(numbered) >= 3:
        gaps = _candidate_line_gaps(numbered)
        if _numbered_run_is_predominantly_content_like(numbered):
            return (
                "local_enumeration",
                SOURCE_STRUCTURE_MEDIUM,
                "Numbered run is predominantly interrogative/prose-like content",
                "content_like_sequence_penalty",
            )
        if dense and textual:
            return (
                "local_enumeration",
                SOURCE_STRUCTURE_MEDIUM,
                "Dense numbered sequence appears immediately under textual heading",
                "local_enumeration_support",
            )
        if dense and not structural_labels:
            return (
                "local_enumeration",
                SOURCE_STRUCTURE_MEDIUM,
                "Compact numbered sequence lacks section-scale separation",
                "compact_sequence_penalty",
            )
        if gaps and max(gaps) >= 2:
            return (
                "structural_outline",
                SOURCE_STRUCTURE_HIGH,
                "Numbered sequence is separated by substantive intervening material",
                "substantive_content_separation_support",
            )

    if textual and len(textual) >= 2:
        return (
            "mixed",
            SOURCE_STRUCTURE_MEDIUM,
            "Region contains repeated textual heading candidates without formal numbering",
            "repeated_heading_pattern",
        )

    if count == 1:
        return (
            "unknown",
            SOURCE_STRUCTURE_LOW,
            "Single isolated candidate has insufficient local region evidence",
            "region_ambiguity_penalty",
        )

    return (
        "unknown",
        SOURCE_STRUCTURE_LOW,
        "Region does not contain enough contextual evidence for a structural hypothesis",
        "region_ambiguity_penalty",
    )


def _classify_numbered_sequences(
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    signals: List[SourceStructureSignal],
) -> List[SourceStructureSequence]:
    sequences: List[SourceStructureSequence] = []

    for region in regions:
        region_candidates = [
            candidate for candidate in candidates
            if candidate.region_id == region.id
            and candidate.numbering
            and candidate.numbering.isdigit()
        ]
        region_candidates.sort(key=lambda c: c.source_order)
        for run in _candidate_number_runs(region_candidates):
            if len(run) < 3:
                continue
            sequence_id = f"seq-{len(sequences) + 1:04d}"
            sequence_type, confidence, reason, signal_type, strength = _classify_sequence_run(run, region)
            signal = _add_region_signal(
                signals,
                sequence_id,
                run[0],
                signal_type,
                strength,
                reason,
            )
            support_ids = [signal.id] if signal.strength > 0 else []
            negative_ids = [signal.id] if signal.strength <= 0 else []
            for candidate in run:
                candidate.sequence_id = sequence_id
            sequences.append(
                SourceStructureSequence(
                    id=sequence_id,
                    sequence_type=sequence_type,
                    confidence=confidence,
                    candidate_node_ids=[candidate.id for candidate in run],
                    region_id=region.id,
                    supporting_signal_ids=support_ids,
                    negative_signal_ids=negative_ids,
                    reason=reason,
                )
            )

    return sequences


def _build_section_scopes(
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    signals: List[SourceStructureSignal],
    global_skeleton: _GlobalNumberingSkeleton,
) -> List[SourceStructureScope]:
    scopes: List[SourceStructureScope] = []
    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    claimed_sequence_ids = set()

    for sequence in sequences:
        if sequence.id in claimed_sequence_ids:
            continue
        if sequence.sequence_type == "procedural_sequence":
            continue
        sequence_candidates = [
            candidates_by_id[candidate_id]
            for candidate_id in sequence.candidate_node_ids
            if candidate_id in candidates_by_id
        ]
        sequence_candidates = [
            candidate
            for candidate in sequence_candidates
            if candidate.id not in global_skeleton.contained_local_integer_ids
        ]
        if len(sequence_candidates) < 3:
            continue
        sequence_candidates.sort(key=lambda c: c.source_order)
        if _numbered_run_is_predominantly_content_like(sequence_candidates):
            continue
        if _sequence_has_explicit_numbering_parent(sequence_candidates, candidates):
            continue

        opener = _find_section_scope_opener(sequence_candidates[0], candidates, regions)
        if not opener:
            continue

        if _looks_like_procedural_scope(opener, sequence_candidates):
            continue

        scope_id = f"scope-{len(scopes) + 1:04d}"
        contained_candidates = [opener] + sequence_candidates
        contained_region_ids = sorted({
            candidate.region_id
            for candidate in contained_candidates
            if candidate.region_id
        })
        pages = [
            candidate.source_page
            for candidate in contained_candidates
            if candidate.source_page is not None
        ]
        signal = _add_region_signal(
            signals,
            scope_id,
            opener,
            "section_scope_support",
            0.82,
            "Semantic heading opens a section scope containing a later coherent numbered sequence",
        )
        opener.status = STATUS_ACCEPTED
        opener.confidence = SOURCE_STRUCTURE_HIGH
        opener.confidence_score = max(opener.confidence_score, 0.84)
        opener.reason = "Semantic heading opens a section scope containing a coherent local numbered sequence"
        opener.supporting_signal_ids.append(signal.id)
        opener.section_scope_id = scope_id
        for candidate in sequence_candidates:
            candidate.section_scope_id = scope_id
            candidate.status = STATUS_ACCEPTED
            candidate.confidence = SOURCE_STRUCTURE_HIGH
            candidate.confidence_score = max(candidate.confidence_score, 0.8)
            candidate.reason = "Numbered item is locally structural inside an evidence-backed section scope"
            candidate.supporting_signal_ids.append(signal.id)
            if candidate.hierarchy_level <= opener.hierarchy_level:
                candidate.hierarchy_level = opener.hierarchy_level + 1

        scopes.append(
            SourceStructureScope(
                id=scope_id,
                opener_node_id=opener.id,
                opener_title=_display_title(opener),
                confidence=SOURCE_STRUCTURE_HIGH,
                start_page=min(pages) if pages else None,
                end_page=max(pages) if pages else None,
                start_block=min(candidate.source_block for candidate in contained_candidates),
                end_block=max(candidate.source_block for candidate in contained_candidates),
                start_line=opener.source_line,
                end_line=sequence_candidates[-1].source_line,
                contained_region_ids=contained_region_ids,
                contained_sequence_ids=[sequence.id],
                supporting_signal_ids=[signal.id],
                reason="Heading and numbered sequence share a source-content span without stronger intervening boundary",
            )
        )
        claimed_sequence_ids.add(sequence.id)

    return scopes


def _assign_structural_scopes(
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    section_scopes: List[SourceStructureScope],
    global_skeleton: _GlobalNumberingSkeleton,
) -> None:
    scoped_sequence_ids = {
        sequence_id
        for section_scope in section_scopes
        for sequence_id in section_scope.contained_sequence_ids
    }
    scope_opener_ids = {
        section_scope.opener_node_id
        for section_scope in section_scopes
    }

    for candidate in candidates:
        if candidate.status != STATUS_ACCEPTED:
            candidate.structural_scope = SCOPE_UNRESOLVED
            continue

        if candidate.id in global_skeleton.contained_local_integer_ids:
            candidate.structural_scope = SCOPE_LOCAL if candidate.section_scope_id else SCOPE_UNRESOLVED
            continue

        region = _region_for_candidate(candidate, regions)
        sequence = _sequence_for_candidate(candidate, sequences)

        if candidate.id in scope_opener_ids:
            candidate.structural_scope = SCOPE_GLOBAL
            continue

        if sequence and sequence.id in scoped_sequence_ids:
            candidate.structural_scope = SCOPE_LOCAL
            continue

        if sequence and sequence.sequence_type == "semantic_local_sequence":
            candidate.structural_scope = SCOPE_LOCAL
            continue

        if region and region.region_type == "semantic_local_scope":
            candidate.structural_scope = (
                SCOPE_GLOBAL
                if candidate.detection_method == "textual"
                else SCOPE_LOCAL
            )
            continue

        if sequence and sequence.sequence_type in {"local_enumeration", "procedural_sequence"}:
            candidate.structural_scope = SCOPE_UNRESOLVED
            continue

        if candidate.detection_method in {
            "academic_numbered",
            "roman",
            "split_roman",
            "structural_label",
            "textual",
        }:
            candidate.structural_scope = SCOPE_GLOBAL
        elif (
            candidate.detection_method in {"numbered", "split_numbered"}
            and _has_positive_global_numbering_evidence(
                candidate,
                candidates,
                regions,
                sequences,
                global_skeleton.contained_local_integer_ids,
            )
        ):
            candidate.structural_scope = SCOPE_GLOBAL
        else:
            candidate.structural_scope = SCOPE_UNRESOLVED


def _construct_hierarchy(
    candidates: List[_Candidate],
    signals: List[SourceStructureSignal],
    regions: List[SourceStructureRegion],
    section_scopes: List[SourceStructureScope],
    global_skeleton: _GlobalNumberingSkeleton,
) -> List[SourceStructureRelationship]:
    relationships: List[SourceStructureRelationship] = []
    accepted = [
        c for c in candidates
        if c.status == STATUS_ACCEPTED
        and c.id not in global_skeleton.contained_local_integer_ids
    ]
    by_numbering = {c.numbering: c for c in accepted if c.numbering}
    stack_by_level: Dict[int, _Candidate] = {}

    for candidate in sorted(accepted, key=lambda c: c.source_order):
        parent = None

        relationship_type = "parent_child"

        if candidate.numbering and "." in candidate.numbering:
            parts = candidate.numbering.split(".")
            for depth in range(len(parts) - 1, 0, -1):
                parent_number = ".".join(parts[:depth])
                parent = by_numbering.get(parent_number)
                if parent:
                    relationship_type = "explicit_numbering_parent"
                    break

        if parent is None:
            parent = stack_by_level.get(candidate.hierarchy_level - 1)
            if parent and parent.detection_method == "textual":
                relationship_type = "semantic_parent"
            elif parent and parent.detection_method == "structural_label":
                relationship_type = "explicit_structural_parent"

        if (
            parent
            and parent.source_order < candidate.source_order
            and _relationship_is_structurally_compatible(parent, candidate, candidates, regions, section_scopes)
        ):
            candidate.parent_id = parent.id
            signal = _add_signal(
                signals,
                candidate,
                "parent_child_relationship_support",
                0.8,
                f"{candidate.id} follows parent {parent.id} in source order and hierarchy level",
            )
            relationships.append(
                SourceStructureRelationship(
                    id=f"rel-{len(relationships) + 1:04d}",
                    parent_node_id=parent.id,
                    child_node_id=candidate.id,
                    relationship_type=relationship_type,
                    confidence=SOURCE_STRUCTURE_HIGH if candidate.confidence == SOURCE_STRUCTURE_HIGH else SOURCE_STRUCTURE_MEDIUM,
                    confidence_score=0.85 if candidate.confidence == SOURCE_STRUCTURE_HIGH else 0.65,
                    supporting_signal_ids=[signal.id],
                    reason="Explicit hierarchy/level relationship",
                )
            )

        stack_by_level[candidate.hierarchy_level] = candidate
        for level in list(stack_by_level):
            if level > candidate.hierarchy_level:
                del stack_by_level[level]

    return relationships


def _evaluate_structure_confidence(
    candidates: List[_Candidate],
    relationships: List[SourceStructureRelationship],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    section_scopes: List[SourceStructureScope],
) -> Tuple[str, str]:
    accepted = [c for c in candidates if c.status == STATUS_ACCEPTED]
    high = [c for c in accepted if c.confidence == SOURCE_STRUCTURE_HIGH]
    medium = [c for c in accepted if c.confidence == SOURCE_STRUCTURE_MEDIUM]
    numbered = [
        c for c in high
        if c.detection_method in {"numbered", "academic_numbered", "split_numbered"}
        and c.structural_scope == SCOPE_GLOBAL
    ]
    explicit_labels = [c for c in high if c.detection_method == "structural_label"]
    high_global_regions = [
        r for r in regions
        if r.confidence == SOURCE_STRUCTURE_HIGH
        and r.region_type in {"structural_outline", "toc_like", "mixed"}
    ]
    high_local_sequences = [
        sequence for sequence in sequences
        if sequence.confidence == SOURCE_STRUCTURE_HIGH
        and (
            sequence.sequence_type == "semantic_local_sequence"
            or any(sequence.id in scope.contained_sequence_ids for scope in section_scopes)
        )
    ]
    high_section_scopes = [
        section_scope for section_scope in section_scopes
        if section_scope.confidence == SOURCE_STRUCTURE_HIGH
    ]
    semantic_relationships = [
        relationship for relationship in relationships
        if relationship.relationship_type == "semantic_parent"
    ]
    local_confidence = (
        SOURCE_STRUCTURE_HIGH
        if (high_local_sequences or high_section_scopes) and semantic_relationships
        else SOURCE_STRUCTURE_MEDIUM
        if any(c.structural_scope == SCOPE_LOCAL for c in high)
        else SOURCE_STRUCTURE_LOW
    )

    if len(numbered) >= 3 and _max_hierarchy_depth(numbered) >= 2 and relationships:
        return SOURCE_STRUCTURE_HIGH, local_confidence

    if len(numbered) >= 4 and _has_coherent_top_level_sequence(numbered) and high_global_regions:
        return SOURCE_STRUCTURE_HIGH, local_confidence

    if explicit_labels and len(accepted) >= 3 and relationships:
        return SOURCE_STRUCTURE_HIGH, local_confidence

    if len(high) >= 2 or len(medium) >= 3:
        return SOURCE_STRUCTURE_MEDIUM, local_confidence

    return SOURCE_STRUCTURE_LOW, local_confidence


def _compute_global_numbering_skeleton(
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    signals: List[SourceStructureSignal],
) -> _GlobalNumberingSkeleton:
    excluded: Dict[str, str] = {}
    contained_ids = _integer_candidates_inside_dotted_subsection_spans(candidates)

    eligible_by_number: Dict[int, List[_Candidate]] = {}
    for candidate in candidates:
        if not _is_global_skeleton_candidate_eligible(candidate, contained_ids):
            if candidate.id in contained_ids:
                reason = "Integer candidate is contained inside an explicit dotted subsection span"
                excluded[candidate.id] = reason
                _add_signal(
                    signals,
                    candidate,
                    "global_numbering_skeleton_exclusion",
                    -0.55,
                    reason,
                )
            continue
        eligible_by_number.setdefault(int(candidate.numbering or "0"), []).append(candidate)

    representatives: Dict[int, _Candidate] = {}
    for number, number_candidates in eligible_by_number.items():
        representative = max(
            number_candidates,
            key=lambda candidate: _global_skeleton_candidate_score(candidate, candidates, regions, sequences),
        )
        representatives[number] = representative

    supported: Dict[str, _SkeletonSupport] = {}
    for run in _consecutive_runs(sorted(representatives)):
        if len(run) < 5:
            continue
        run_candidates = [representatives[number] for number in run]
        anchor_count = sum(
            1 for candidate in run_candidates
            if _global_skeleton_anchor_score(candidate, candidates, regions, sequences) > 0
        )
        if anchor_count < 2:
            continue

        for candidate in run_candidates:
            anchor = _global_skeleton_anchor_score(candidate, candidates, regions, sequences)
            confidence = SOURCE_STRUCTURE_HIGH if anchor > 0 or len(run) >= 6 else SOURCE_STRUCTURE_MEDIUM
            score = 0.84 if confidence == SOURCE_STRUCTURE_HIGH else 0.72
            reason = (
                f"Top-level heading belongs to coherent global numbering skeleton "
                f"{run[0]}–{run[-1]} with {anchor_count} anchored members"
            )
            supported[candidate.id] = _SkeletonSupport(confidence, score, reason)
            _add_signal(
                signals,
                candidate,
                "global_numbering_skeleton_support",
                score,
                reason,
            )

    return _GlobalNumberingSkeleton(
        supported=supported,
        excluded=excluded,
        contained_local_integer_ids=contained_ids,
    )


def _is_global_skeleton_candidate_eligible(
    candidate: _Candidate,
    contained_ids: set,
) -> bool:
    if candidate.id in contained_ids:
        return False
    if candidate.detection_method == "procedural_numbered_sentence":
        return False
    if not candidate.numbering or not candidate.numbering.isdigit():
        return False
    if not _is_safe_numbering(candidate.numbering):
        return False
    if _strong_heading_title_safety_issue(candidate.title):
        return False
    if _numbered_candidate_is_content_like(candidate):
        return False
    if _looks_like_non_rehabilitatable_numbered_list_item(candidate):
        return False

    return True


def _global_skeleton_candidate_score(
    candidate: _Candidate,
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
) -> Tuple[int, int, int, int]:
    return (
        _global_skeleton_anchor_score(candidate, candidates, regions, sequences),
        1 if _title_shape_is_strong(candidate.title) else 0,
        -candidate.source_order,
        -candidate.source_line,
    )


def _global_skeleton_anchor_score(
    candidate: _Candidate,
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
) -> int:
    score = 0
    direct_children = [
        possible for possible in candidates
        if _direct_parent_numbering(possible.numbering) == candidate.numbering
        and possible.source_order > candidate.source_order
        and _is_safe_numbering(possible.numbering or "")
    ]
    if direct_children:
        score += 2

    region = _region_for_candidate(candidate, regions)
    sequence = _sequence_for_candidate(candidate, sequences)
    if region and region.region_type in {"structural_outline", "toc_like"}:
        score += 1
    if sequence and sequence.sequence_type in {"structural_sequence", "toc_sequence"}:
        score += 1
    return score


def _integer_candidates_inside_dotted_subsection_spans(candidates: List[_Candidate]) -> set:
    contained = set()
    ordered = sorted(candidates, key=lambda c: c.source_order)
    dotted = [
        candidate for candidate in ordered
        if candidate.numbering and "." in candidate.numbering
    ]
    for opener in dotted:
        boundary = _next_dotted_sibling_or_boundary(opener, ordered)
        if not boundary:
            continue
        for candidate in ordered:
            if candidate.source_order <= opener.source_order:
                continue
            if candidate.source_order >= boundary.source_order:
                continue
            if candidate.numbering and candidate.numbering.isdigit():
                contained.add(candidate.id)
    return contained


def _next_dotted_sibling_or_boundary(
    opener: _Candidate,
    ordered_candidates: List[_Candidate],
) -> Optional[_Candidate]:
    opener_parent = _direct_parent_numbering(opener.numbering)
    opener_level = opener.hierarchy_level
    for candidate in ordered_candidates:
        if candidate.source_order <= opener.source_order:
            continue
        if not candidate.numbering or "." not in candidate.numbering:
            continue
        if (
            candidate.hierarchy_level == opener_level
            and _direct_parent_numbering(candidate.numbering) == opener_parent
        ):
            return candidate
    return None


def _validate_numbered_candidate(
    candidate: _Candidate,
    candidates: List[_Candidate],
    signals: List[SourceStructureSignal],
    numbered_groups: Dict[str, List[_Candidate]],
    top_level_numbers: List[int],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    global_skeleton: _GlobalNumberingSkeleton,
) -> None:
    if not candidate.numbering or not _is_safe_numbering(candidate.numbering):
        _reject(candidate, signals, "unsafe_numbering_penalty", "Numbering is outside safe academic heading bounds")
        return

    if (
        _looks_like_numbered_list_item(candidate)
        and candidate.id not in global_skeleton.supported
    ):
        _reject(candidate, signals, "numbered_list_penalty", "Numbered candidate looks like an ordinary list item")
        return

    if candidate.id in global_skeleton.excluded:
        _uncertain(
            candidate,
            signals,
            "global_numbering_skeleton_exclusion",
            global_skeleton.excluded[candidate.id],
        )
        return

    region = _region_for_candidate(candidate, regions)
    sequence = _sequence_for_candidate(candidate, sequences)

    if sequence and sequence.sequence_type == "procedural_sequence":
        _reject(candidate, signals, "procedural_sequence_support", "Candidate belongs to a procedural numbered sequence")
        return

    if sequence and sequence.sequence_type == "local_enumeration":
        _uncertain(candidate, signals, "local_enumeration_support", "Candidate belongs to a dense local enumeration, not a document-level section")
        return

    child_count = len(numbered_groups.get(candidate.numbering, []))
    if child_count >= 2:
        _accept(
            candidate,
            signals,
            SOURCE_STRUCTURE_HIGH,
            0.96,
            "child_numbering_support",
            f"Numbered heading has {child_count} numbered children",
        )
        return

    if "." in candidate.numbering:
        parent = ".".join(candidate.numbering.split(".")[:-1])
        if any(c.numbering == parent for c in candidates):
            _accept(
                candidate,
                signals,
                SOURCE_STRUCTURE_HIGH,
                0.9,
                "coherent_numbered_hierarchy",
                f"Subsection belongs to detected parent {parent}",
            )
        else:
            sibling_count = len(numbered_groups.get(parent, []))
            if sibling_count >= 2:
                _accept(
                    candidate,
                    signals,
                    SOURCE_STRUCTURE_MEDIUM,
                    0.7,
                    "sibling_numbering_support",
                    f"Subsection is part of a repeated sibling sequence under {parent}",
                )
            else:
                _uncertain(candidate, signals, "orphan_subsection", "Numbered subsection lacks detected parent/sibling support")
        return

    number_value = int(candidate.numbering)
    if sequence and sequence.sequence_type == "structural_sequence":
        _accept(
            candidate,
            signals,
            SOURCE_STRUCTURE_HIGH,
            0.86,
            "substantive_content_separation_support",
            "Top-level heading belongs to a context-supported structural sequence",
        )
        return

    if sequence and sequence.sequence_type == "toc_sequence":
        _accept(
            candidate,
            signals,
            SOURCE_STRUCTURE_HIGH,
            0.84,
            "toc_density_support",
            "Top-level entry belongs to a TOC/outline-like structural region",
        )
        return

    if sequence and sequence.sequence_type == "semantic_local_sequence":
        _accept(
            candidate,
            signals,
            SOURCE_STRUCTURE_HIGH,
            0.82,
            "semantic_local_scope_support",
            "Numbered candidate belongs to a coherent local sequence under a semantic heading",
        )
        return

    if (
        _number_has_sequence_support(number_value, top_level_numbers)
        and region
        and region.region_type in {"structural_outline", "toc_like"}
    ):
        _accept(
            candidate,
            signals,
            SOURCE_STRUCTURE_MEDIUM,
            0.68,
            "structural_region_support",
            "Top-level heading has sequence support inside a structural region",
        )
        return

    skeleton_support = global_skeleton.supported.get(candidate.id)
    if skeleton_support:
        _accept(
            candidate,
            signals,
            skeleton_support.confidence,
            skeleton_support.score,
            "global_numbering_skeleton_support",
            skeleton_support.reason,
        )
        return

    if _title_shape_is_strong(candidate.title):
        _uncertain(
            candidate,
            signals,
            "isolated_candidate_penalty",
            "Plausible numbered heading but no child or sequence support yet",
        )
    else:
        _reject(candidate, signals, "isolated_candidate_penalty", "Isolated weak numbered candidate")


def _rehabilitate_explicit_numbering_parents(
    candidates: List[_Candidate],
    signals: List[SourceStructureSignal],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    global_skeleton: _GlobalNumberingSkeleton,
) -> None:
    """Restore weak explicit parents proven by a safe descendant skeleton.

    This is intentionally narrower than normal validation.  It only handles
    exact numbered ancestry such as 15 -> 15.1/15.2/15.3, and it never
    reopens candidates rejected by strong false-positive protections.
    """

    for candidate in candidates:
        if candidate.status == STATUS_ACCEPTED:
            continue
        if candidate.id in global_skeleton.contained_local_integer_ids:
            continue
        if not _candidate_can_be_rehabilitated_by_explicit_descendants(
            candidate,
            regions,
            sequences,
        ):
            continue

        descendants = _safe_explicit_descendants_for_parent(
            candidate,
            candidates,
            regions,
            sequences,
        )
        direct_descendants = [
            descendant
            for descendant in descendants
            if _direct_parent_numbering(descendant.numbering) == candidate.numbering
        ]
        skeleton_support = global_skeleton.supported.get(candidate.id)
        if len(direct_descendants) < 2 and not (
            len(direct_descendants) == 1
            and skeleton_support
        ):
            continue

        confidence = (
            SOURCE_STRUCTURE_HIGH
            if len(direct_descendants) >= 3
            else skeleton_support.confidence
            if skeleton_support
            else SOURCE_STRUCTURE_MEDIUM
        )
        score = (
            0.88
            if confidence == SOURCE_STRUCTURE_HIGH
            else skeleton_support.score
            if skeleton_support
            else 0.72
        )
        candidate.status = STATUS_ACCEPTED
        candidate.confidence = confidence
        candidate.confidence_score = max(candidate.confidence_score, score)
        candidate.reason = (
            f"Explicit numbered parent restored by {len(direct_descendants)} "
            "accepted safe descendants"
            + (" and global numbering skeleton support" if skeleton_support else "")
        )
        signal = _add_signal(
            signals,
            candidate,
            "explicit_descendant_support",
            score,
            (
                f"Candidate {candidate.numbering} is a plausible numbered parent "
                f"for {', '.join(descendant.numbering or descendant.id for descendant in direct_descendants)}"
            ),
        )
        candidate.supporting_signal_ids.append(signal.id)


def _candidate_can_be_rehabilitated_by_explicit_descendants(
    candidate: _Candidate,
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
) -> bool:
    if candidate.status not in {STATUS_REJECTED, STATUS_UNCERTAIN}:
        return False
    if not candidate.numbering or "." in candidate.numbering:
        return False
    if not _is_safe_numbering(candidate.numbering):
        return False
    if _strong_heading_title_safety_issue(candidate.title):
        return False
    if _looks_like_non_rehabilitatable_numbered_list_item(candidate):
        return False

    return _title_shape_is_strong(candidate.title) or _weak_heading_title_safety_issue(candidate.title)


def _safe_explicit_descendants_for_parent(
    parent: _Candidate,
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
) -> List[_Candidate]:
    descendants = []
    for candidate in candidates:
        if candidate.status != STATUS_ACCEPTED:
            continue
        if not _is_exact_numbering_descendant(parent.numbering, candidate.numbering):
            continue
        if candidate.source_order <= parent.source_order:
            continue
        if not _is_safe_numbering(candidate.numbering or ""):
            continue
        if _strong_heading_title_safety_issue(candidate.title):
            continue
        if _looks_like_non_rehabilitatable_numbered_list_item(candidate):
            continue
        descendants.append(candidate)
    return descendants


def _is_exact_numbering_descendant(
    parent_numbering: Optional[str],
    child_numbering: Optional[str],
) -> bool:
    return bool(
        parent_numbering
        and child_numbering
        and "." in child_numbering
        and child_numbering.startswith(parent_numbering + ".")
    )


def _direct_parent_numbering(numbering: Optional[str]) -> Optional[str]:
    if not numbering or "." not in numbering:
        return None
    return ".".join(numbering.split(".")[:-1])


def _looks_like_non_rehabilitatable_numbered_list_item(candidate: _Candidate) -> bool:
    title = candidate.title.strip()
    if _SENTENCE_START_RE.match(title):
        return True
    if len(title.split()) > 12 and title.endswith("."):
        return True
    if _LOWERCASE_START_RE.match(title) and not _starts_with_scientific_lowercase_acronym(title):
        return True
    if candidate.numbering and candidate.numbering.isdigit():
        try:
            if int(candidate.numbering) > _MAX_PROCEDURAL_NUMBER:
                return True
        except ValueError:
            return False
    return False


def _starts_with_scientific_lowercase_acronym(title: str) -> bool:
    first_word = (title.strip().split() or [""])[0]
    return bool(re.match(r"^(?:[a-z][A-Z]{1,4}|[a-z]{1,3}RNA|[a-z]{1,3}DNA)\b", first_word))


def _validate_roman_candidate(
    candidate: _Candidate,
    signals: List[SourceStructureSignal],
    roman_sequence: List[int],
) -> None:
    value = _roman_to_int(candidate.numbering or "")
    if value <= 0 or value > 30:
        _reject(candidate, signals, "unsafe_roman_numbering_penalty", "Roman numeral is outside safe structural range")
        return

    if _number_has_sequence_support(value, roman_sequence):
        _accept(candidate, signals, SOURCE_STRUCTURE_MEDIUM, 0.72, "roman_sequence_support", "Roman heading appears in a coherent sequence")
    else:
        _uncertain(candidate, signals, "isolated_roman_candidate", "Roman heading lacks sequence support")


def _validate_textual_candidate(
    candidate: _Candidate,
    signals: List[SourceStructureSignal],
    textual_patterns: Dict[str, int],
    structural_label_context: bool,
    regions: List[SourceStructureRegion],
) -> None:
    region = _region_for_candidate(candidate, regions)
    if region and region.region_type == "semantic_local_scope":
        _accept(
            candidate,
            signals,
            SOURCE_STRUCTURE_HIGH,
            0.82,
            "semantic_local_parent_support",
            "Textual heading acts as semantic parent for a coherent local numbered sequence",
        )
        candidate.structural_scope = SCOPE_GLOBAL
        return

    pattern = _textual_pattern_key(candidate.title)
    if textual_patterns.get(pattern, 0) >= 2:
        _accept(candidate, signals, SOURCE_STRUCTURE_MEDIUM, 0.68, "repeated_heading_pattern", "Textual heading shape repeats in the document")
        return

    if structural_label_context and _title_shape_is_strong(candidate.title):
        _uncertain(candidate, signals, "textual_heading_shape", "Textual heading is plausible but lacks repeated pattern support")
        return

    _uncertain(candidate, signals, "isolated_candidate_penalty", "Isolated textual heading candidate")


def _extract_procedural_numbered_sentence(
    line: str,
    page: Optional[int],
    source_block: int,
    source_line: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _PROCEDURAL_DOTTED_SENTENCE_RE.match(line)
    if not match:
        return None

    return _candidate(
        raw_text=line,
        title=match.group("sentence"),
        numbering=match.group("number"),
        hierarchy_level=1,
        page=page,
        block=source_block,
        line=source_line,
        order=source_order,
        method="procedural_numbered_sentence",
    )


def _extract_numbered_heading(
    line: str,
    page: Optional[int],
    source_block: int,
    source_line: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _NUMBERED_HEADING_RE.match(line)
    if not match:
        return None
    numbering = match.group("number")
    title = _clean_heading_title(match.group("title"))
    return _candidate(line, title, numbering, _heading_level_from_numbering(numbering), page, source_block, source_line, source_order, "numbered")


def _extract_academic_dotted_heading(
    line: str,
    page: Optional[int],
    source_block: int,
    source_line: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _ACADEMIC_DOTTED_TOP_LEVEL_RE.match(line)
    if not match:
        return None

    number = match.group("number")
    title = _clean_heading_title(match.group("title"))
    return _candidate(line, title, number, 1, page, source_block, source_line, source_order, "academic_numbered")


def _extract_split_numbered_heading(
    line: str,
    lines: List[str],
    line_index: int,
    page: Optional[int],
    source_block: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _STANDALONE_SECTION_NUMBER_RE.match(line)
    if not match:
        return None

    numbering = match.group("number")
    if not _is_safe_standalone_section_number(numbering):
        return None

    next_line_index, next_line = _next_non_empty_line(lines, line_index)
    if next_line_index is None or not next_line:
        return None

    if (
        _NUMBERED_HEADING_RE.match(next_line)
        or _ACADEMIC_DOTTED_TOP_LEVEL_RE.match(next_line)
    ):
        return None

    title = _clean_heading_title(next_line)
    return _candidate(f"{numbering} {title}", title, numbering, 1, page, source_block, line_index, source_order, "split_numbered")


def _extract_roman_heading(
    line: str,
    page: Optional[int],
    source_block: int,
    source_line: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _ROMAN_RE.match(line)
    if not match:
        return None
    roman = match.group("roman").upper().rstrip(".")
    title = _clean_heading_title(match.group("title"))
    return _candidate(line, title, roman, 1, page, source_block, source_line, source_order, "roman")


def _extract_split_roman_heading(
    line: str,
    lines: List[str],
    line_index: int,
    page: Optional[int],
    source_block: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _ROMAN_STANDALONE_RE.match(line)
    if not match:
        return None

    roman = match.group("roman").upper().rstrip(".")
    next_line_index, next_line = _next_non_empty_line(lines, line_index)
    if next_line_index is None or not next_line:
        return None
    title = _clean_heading_title(next_line)
    return _candidate(f"{roman} {title}", title, roman, 1, page, source_block, line_index, source_order, "split_roman")


def _extract_structural_label(
    line: str,
    lines: List[str],
    line_index: int,
    page: Optional[int],
    source_block: int,
    source_order: int,
) -> Optional[_Candidate]:
    match = _STRUCTURAL_LABEL_RE.match(line)
    if not match:
        return None

    label = match.group("label")
    rest = _clean_heading_title(match.group("rest"))
    original_lines = [line]
    consumed_lines = 1
    next_line_index, next_line = _next_non_empty_line(lines, line_index)
    if (
        next_line_index == line_index + 1
        and next_line
        and not _looks_like_conflicting_heading_start(next_line)
        and _is_structural_continuation_line(next_line)
    ):
        rest = f"{rest} {next_line}".strip()
        original_lines.append(next_line)
        consumed_lines = 2

    title = f"{label.upper()} {rest}"
    level = _LABEL_LEVELS.get(label.lower(), 1)
    candidate = _candidate(line, title, None, level, page, source_block, line_index, source_order, "structural_label")
    candidate.original_lines = original_lines
    candidate.consumed_lines = consumed_lines
    return candidate


def _extract_textual_heading(
    line: str,
    lines: List[str],
    line_index: int,
    page: Optional[int],
    source_block: int,
    source_order: int,
) -> Optional[_Candidate]:
    if not _is_potential_textual_heading(line):
        return None

    previous_blank = line_index <= 1 or not lines[line_index - 2].strip()
    next_blank = line_index >= len(lines) or not lines[line_index].strip()
    if not (previous_blank or next_blank):
        return None

    return _candidate(line, line, None, 1, page, source_block, line_index, source_order, "textual")


def _candidate(
    raw_text: str,
    title: str,
    numbering: Optional[str],
    hierarchy_level: int,
    page: Optional[int],
    block: int,
    line: int,
    order: int,
    method: str,
) -> _Candidate:
    return _Candidate(
        id=f"node-{order + 1:04d}",
        raw_text=raw_text,
        title=_clean_heading_title(title),
        normalized_title=_normalize_heading_title(title),
        numbering=numbering,
        hierarchy_level=hierarchy_level,
        source_page=page,
        source_block=block,
        source_line=line,
        source_order=order,
        detection_method=method,
        original_lines=[raw_text],
    )


def _accept(
    candidate: _Candidate,
    signals: List[SourceStructureSignal],
    confidence: str,
    score: float,
    signal_type: str,
    reason: str,
) -> None:
    candidate.status = STATUS_ACCEPTED
    candidate.confidence = confidence
    candidate.confidence_score = score
    candidate.reason = reason
    _add_signal(signals, candidate, signal_type, score, reason)


def _uncertain(
    candidate: _Candidate,
    signals: List[SourceStructureSignal],
    signal_type: str,
    reason: str,
) -> None:
    candidate.status = STATUS_UNCERTAIN
    candidate.confidence = SOURCE_STRUCTURE_LOW
    candidate.confidence_score = max(candidate.confidence_score, 0.38)
    candidate.reason = reason
    signal = _add_signal(signals, candidate, signal_type, -0.35, reason)
    candidate.negative_signal_ids.append(signal.id)


def _reject(
    candidate: _Candidate,
    signals: List[SourceStructureSignal],
    signal_type: str,
    reason: str,
) -> None:
    candidate.status = STATUS_REJECTED
    candidate.confidence = SOURCE_STRUCTURE_LOW
    candidate.confidence_score = 0.0
    candidate.reason = reason
    signal = _add_signal(signals, candidate, signal_type, -0.85, reason)
    candidate.negative_signal_ids.append(signal.id)


def _add_signal(
    signals: List[SourceStructureSignal],
    candidate: _Candidate,
    signal_type: str,
    strength: float,
    explanation: str,
) -> SourceStructureSignal:
    signal = SourceStructureSignal(
        id=f"sig-{len(signals) + 1:04d}",
        signal_type=signal_type,
        source_page=candidate.source_page,
        source_block=candidate.source_block,
        source_line=candidate.source_line,
        strength=round(strength, 3),
        explanation=explanation,
        target_id=candidate.id,
    )
    signals.append(signal)
    if strength > 0:
        candidate.supporting_signal_ids.append(signal.id)
    return signal


def _add_region_signal(
    signals: List[SourceStructureSignal],
    target_id: str,
    candidate: _Candidate,
    signal_type: str,
    strength: float,
    explanation: str,
) -> SourceStructureSignal:
    signal = SourceStructureSignal(
        id=f"sig-{len(signals) + 1:04d}",
        signal_type=signal_type,
        source_page=candidate.source_page,
        source_block=candidate.source_block,
        source_line=candidate.source_line,
        strength=round(strength, 3),
        explanation=explanation,
        target_id=target_id,
    )
    signals.append(signal)
    return signal


def _initial_signal_for_candidate(candidate: _Candidate) -> str:
    return {
        "numbered": "arabic_numbering_shape",
        "academic_numbered": "academic_top_level_numbering_shape",
        "split_numbered": "split_level_one_numbering_shape",
        "roman": "roman_numbering_shape",
        "split_roman": "split_roman_numbering_shape",
        "structural_label": "explicit_structural_label_shape",
        "textual": "textual_heading_shape",
        "procedural_numbered_sentence": "procedural_numbering_shape",
    }.get(candidate.detection_method, "candidate_shape")


def _initial_strength_for_candidate(candidate: _Candidate) -> float:
    if candidate.detection_method == "procedural_numbered_sentence":
        return -0.65
    if candidate.detection_method == "structural_label":
        return 0.7
    if candidate.detection_method in {"numbered", "academic_numbered", "split_numbered"}:
        return 0.55
    if candidate.detection_method in {"roman", "split_roman"}:
        return 0.45
    return 0.35


def _candidate_to_heading(candidate: _Candidate) -> SourceHeading:
    return SourceHeading(
        text=_display_title(candidate),
        normalized_title=candidate.normalized_title,
        numbering=candidate.numbering,
        hierarchy_level=candidate.hierarchy_level,
        source_page=candidate.source_page,
        source_block=candidate.source_block,
        source_line=candidate.source_line,
        document_order=candidate.source_order,
        detection_method=_legacy_method(candidate.detection_method),
        region_id=candidate.region_id,
        sequence_id=candidate.sequence_id,
        section_scope_id=candidate.section_scope_id,
        structural_scope=candidate.structural_scope,
        status=candidate.status,
        confidence=candidate.confidence,
        confidence_score=round(candidate.confidence_score, 3),
        parent_id=candidate.parent_id,
        node_id=candidate.id,
        reason=candidate.reason,
    )


def _legacy_method(method: str) -> str:
    if method in {"academic_numbered", "split_numbered"}:
        return "numbered"
    if method == "split_roman":
        return "roman"
    return method


def _display_title(candidate: _Candidate) -> str:
    if candidate.numbering and candidate.detection_method not in {"roman", "split_roman"}:
        return f"{candidate.numbering} {candidate.title}"
    if candidate.numbering and candidate.detection_method in {"roman", "split_roman"}:
        return f"{candidate.numbering} {candidate.title}"
    return candidate.title


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _clean_heading_title(title: str) -> str:
    return (title or "").strip().strip("-–—").strip()


def _normalize_heading_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip().lower()


def _heading_level_from_numbering(numbering: str) -> int:
    return numbering.count(".") + 1


def _is_safe_numbering(numbering: str) -> bool:
    if not numbering:
        return False
    first_part = numbering.split(".")[0]
    if not first_part.isdigit():
        return False
    first_number = int(first_part)
    if first_number <= 0 or first_number > _MAX_TOP_LEVEL_SECTION:
        return False
    if "." not in numbering and _YEAR_RE.match(numbering):
        return False
    return True


def _is_safe_standalone_section_number(numbering: str) -> bool:
    return bool(numbering and numbering.isdigit() and 1 <= int(numbering) <= _MAX_TOP_LEVEL_SECTION)


def _looks_like_conflicting_heading_start(line: str) -> bool:
    return bool(
        _NUMBERED_HEADING_RE.match(line)
        or _ACADEMIC_DOTTED_TOP_LEVEL_RE.match(line)
        or _STRUCTURAL_LABEL_RE.match(line)
        or _ROMAN_RE.match(line)
    )


def _is_structural_continuation_line(line: str) -> bool:
    if not line or len(line) > 90:
        return False
    if line.endswith((".", ",", ";")):
        return False
    if not _is_safe_heading_title(line):
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    title_case_words = sum(
        1 for word in line.split()
        if word[:1].isupper() and any(char.isalpha() for char in word)
    )
    return uppercase_ratio >= 0.55 or title_case_words >= max(2, len(line.split()) - 1)


def _next_non_empty_line(lines: List[str], line_index: int):
    for candidate_index in range(line_index + 1, len(lines) + 1):
        candidate = _normalize_line(lines[candidate_index - 1])
        if candidate:
            return candidate_index, candidate
    return None, None


def _is_potential_textual_heading(line: str) -> bool:
    if len(line) < 4 or len(line) > 90:
        return False
    words = line.split()
    if len(words) > _TEXTUAL_HEADING_MAX_WORDS:
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    if (
        _NUMBERED_HEADING_RE.match(line)
        or _ACADEMIC_DOTTED_TOP_LEVEL_RE.match(line)
        or _STRUCTURAL_LABEL_RE.match(line)
    ):
        return False
    if not _is_safe_heading_title(line):
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    title_case_words = sum(
        1 for word in words
        if word[:1].isupper() and any(char.isalpha() for char in word)
    )
    return uppercase_ratio >= 0.65 or title_case_words >= max(2, len(words) - 1)


def _is_safe_heading_title(title: str) -> bool:
    return not (
        _strong_heading_title_safety_issue(title)
        or _weak_heading_title_safety_issue(title)
    )


def _strong_heading_title_safety_issue(title: str) -> bool:
    stripped = (title or "").strip()
    if not stripped:
        return True
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if _MOSTLY_PUNCTUATION_RE.match(stripped):
        return True
    if _FORMULA_MARKERS_RE.search(stripped):
        return True
    formula_marker_count = sum(1 for char in stripped if char in "=+*/^<>")
    if formula_marker_count >= 2:
        return True
    if _CITATION_LINE_RE.search(stripped):
        return True
    if _SENTENCE_START_RE.match(stripped):
        return True
    return False


def _weak_heading_title_safety_issue(title: str) -> bool:
    stripped = (title or "").strip()
    if _strong_heading_title_safety_issue(stripped):
        return False
    words = stripped.split()
    if len(words) > 16:
        return True
    if stripped.endswith((".", ",", ";")):
        return True
    first_alpha = next((char for char in stripped if char.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return True
    return False


def _looks_like_numbered_list_item(candidate: _Candidate) -> bool:
    title = candidate.title.strip()
    if _LOWERCASE_START_RE.match(title):
        return True
    if _SENTENCE_START_RE.match(title):
        return True
    if len(title.split()) > 12 and title.endswith("."):
        return True
    if candidate.numbering and candidate.numbering.isdigit():
        try:
            if int(candidate.numbering) > _MAX_PROCEDURAL_NUMBER:
                return True
        except ValueError:
            return False
    return False


def _title_shape_is_strong(title: str) -> bool:
    words = title.split()
    if 2 <= len(words) <= 10 and not title.endswith("."):
        return True
    letters = [char for char in title if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio >= 0.55


def _numbered_child_groups(
    candidates: Iterable[_Candidate],
    contained_local_integer_ids: Optional[set] = None,
) -> Dict[str, List[_Candidate]]:
    contained_local_integer_ids = contained_local_integer_ids or set()
    groups: Dict[str, List[_Candidate]] = {}
    for candidate in candidates:
        if candidate.id in contained_local_integer_ids:
            continue
        if not candidate.numbering or "." not in candidate.numbering:
            continue
        parent = ".".join(candidate.numbering.split(".")[:-1])
        groups.setdefault(parent, []).append(candidate)
    return groups


def _region_for_candidate(
    candidate: _Candidate,
    regions: Iterable[SourceStructureRegion],
) -> Optional[SourceStructureRegion]:
    for region in regions:
        if candidate.id in region.candidate_node_ids:
            return region
    return None


def _sequence_for_candidate(
    candidate: _Candidate,
    sequences: Iterable[SourceStructureSequence],
) -> Optional[SourceStructureSequence]:
    for sequence in sequences:
        if candidate.id in sequence.candidate_node_ids:
            return sequence
    return None


def _region_signal_strength(confidence: str, region_type: str) -> float:
    if region_type in {"local_enumeration", "procedural", "unknown"}:
        return -0.65 if confidence == SOURCE_STRUCTURE_LOW else -0.45
    if confidence == SOURCE_STRUCTURE_HIGH:
        return 0.75
    if confidence == SOURCE_STRUCTURE_MEDIUM:
        return 0.5
    return -0.35


def _sequence_has_explicit_numbering_parent(
    sequence_candidates: List[_Candidate],
    all_candidates: List[_Candidate],
) -> bool:
    for candidate in sequence_candidates:
        if not candidate.numbering or "." not in candidate.numbering:
            continue
        parent_number = ".".join(candidate.numbering.split(".")[:-1])
        if any(
            possible_parent.numbering == parent_number
            and possible_parent.source_order < candidate.source_order
            for possible_parent in all_candidates
        ):
            return True
    return False


def _find_section_scope_opener(
    first_sequence_candidate: _Candidate,
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
) -> Optional[_Candidate]:
    previous_candidates = [
        candidate for candidate in candidates
        if candidate.source_order < first_sequence_candidate.source_order
    ]
    for candidate in reversed(previous_candidates):
        if _candidate_is_strong_scope_boundary(candidate) and candidate.detection_method != "textual":
            return None
        if candidate.detection_method != "textual":
            continue
        if not _title_shape_is_strong(candidate.title):
            continue
        if _looks_like_procedural_heading_title(candidate.title):
            return None
        if not _source_continuity_supports_scope(candidate, first_sequence_candidate):
            return None
        if _has_intervening_stronger_boundary(candidate, first_sequence_candidate, candidates, regions):
            return None
        return candidate
    return None


def _candidate_is_strong_scope_boundary(candidate: _Candidate) -> bool:
    if candidate.detection_method == "structural_label":
        return True
    if (
        candidate.status == STATUS_ACCEPTED
        and candidate.numbering
        and "." in candidate.numbering
    ):
        return True
    if (
        candidate.status == STATUS_ACCEPTED
        and candidate.structural_scope == SCOPE_GLOBAL
        and candidate.numbering
    ):
        return True
    return False


def _source_continuity_supports_scope(
    opener: _Candidate,
    first_sequence_candidate: _Candidate,
) -> bool:
    if first_sequence_candidate.source_order <= opener.source_order:
        return False
    if first_sequence_candidate.source_order - opener.source_order > 10:
        return False
    if opener.source_block == first_sequence_candidate.source_block:
        return first_sequence_candidate.source_line - opener.source_line <= 80
    if first_sequence_candidate.source_block - opener.source_block <= 2:
        return True
    if (
        opener.source_page is not None
        and first_sequence_candidate.source_page is not None
        and 0 <= first_sequence_candidate.source_page - opener.source_page <= 1
    ):
        return True
    return False


def _has_intervening_stronger_boundary(
    opener: _Candidate,
    first_sequence_candidate: _Candidate,
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
) -> bool:
    for candidate in candidates:
        if not (opener.source_order < candidate.source_order < first_sequence_candidate.source_order):
            continue
        if candidate.detection_method == "textual" and _title_shape_is_strong(candidate.title):
            return True
        if candidate.detection_method == "structural_label":
            return True
        region = _region_for_candidate(candidate, regions)
        if region and region.region_type in {"procedural", "toc_like"}:
            return True
    return False


def _looks_like_procedural_scope(
    opener: _Candidate,
    sequence_candidates: List[_Candidate],
) -> bool:
    if _looks_like_procedural_heading_title(opener.title):
        return True
    return _looks_like_procedural_region([opener] + sequence_candidates)


def _looks_like_procedural_heading_title(title: str) -> bool:
    normalized = _normalize_heading_title(title)
    return (
        normalized in {"procedure", "procedura", "protocol", "protocollo", "method", "methods", "metodo"}
        or "laboratory steps" in normalized
        or "review prompts" in normalized
        or "procedure" in normalized
        or "procedura" in normalized
    )


def _same_section_scope(
    parent: _Candidate,
    child: _Candidate,
    section_scopes: Iterable[SourceStructureScope],
) -> bool:
    if not parent.section_scope_id or parent.section_scope_id != child.section_scope_id:
        return False
    return any(section_scope.id == parent.section_scope_id for section_scope in section_scopes)


def _has_positive_global_numbering_evidence(
    candidate: _Candidate,
    candidates: List[_Candidate],
    regions: List[SourceStructureRegion],
    sequences: List[SourceStructureSequence],
    contained_local_integer_ids: Optional[set] = None,
) -> bool:
    contained_local_integer_ids = contained_local_integer_ids or set()
    if candidate.id in contained_local_integer_ids:
        return False

    if candidate.numbering and "." in candidate.numbering:
        parent_number = ".".join(candidate.numbering.split(".")[:-1])
        if any(
            possible_parent.numbering == parent_number
            and possible_parent.id not in contained_local_integer_ids
            for possible_parent in candidates
        ):
            return True

    child_groups = _numbered_child_groups(candidates, contained_local_integer_ids)
    if candidate.numbering and len(child_groups.get(candidate.numbering, [])) >= 2:
        return True

    region = _region_for_candidate(candidate, regions)
    sequence = _sequence_for_candidate(candidate, sequences)
    if (
        region
        and sequence
        and sequence.sequence_type == "structural_sequence"
        and region.region_type == "structural_outline"
    ):
        region_candidates = [
            possible for possible in candidates
            if possible.region_id == region.id
        ]
        if not any(possible.detection_method == "textual" for possible in region_candidates):
            return True

    return False


def _is_dense_region(candidates: List[_Candidate]) -> bool:
    if len(candidates) < 3:
        return False
    gaps = _candidate_line_gaps(candidates)
    if not gaps:
        return False
    return max(gaps) <= 1 and sum(gaps) / len(gaps) <= 1.2


def _numbered_run_is_predominantly_content_like(candidates: List[_Candidate]) -> bool:
    numbered = [
        candidate for candidate in candidates
        if candidate.numbering and candidate.numbering.isdigit()
    ]
    if len(numbered) < 3:
        return False

    content_like = sum(
        1 for candidate in numbered
        if _numbered_candidate_is_content_like(candidate)
    )
    return content_like / len(numbered) >= 0.6


def _numbered_candidate_is_content_like(candidate: _Candidate) -> bool:
    title = candidate.title.strip()
    normalized = _normalize_heading_title(title)
    if not title:
        return True
    if title.endswith("?") or _INTERROGATIVE_START_RE.match(normalized):
        return True

    evidence = 0
    if title.endswith((".", ",", ";")):
        evidence += 1
    if _PROSE_CONTINUATION_START_RE.match(normalized):
        evidence += 1
    if _strong_heading_title_safety_issue(title) or _weak_heading_title_safety_issue(title):
        evidence += 1
    if _looks_like_non_rehabilitatable_numbered_list_item(candidate):
        evidence += 1
    if len(title.split()) >= 11:
        evidence += 1
    if _CITATION_LINE_RE.search(title):
        evidence += 1

    return evidence >= 2


def _looks_like_procedural_region(candidates: List[_Candidate]) -> bool:
    textual_titles = [
        c.title.lower()
        for c in candidates
        if c.detection_method == "textual"
    ]
    if any(
        title in {"procedure", "procedura", "protocol", "protocollo", "method", "methods", "metodo"}
        or "laboratory steps" in title
        or "review prompts" in title
        for title in textual_titles
    ):
        return True

    numbered = [
        c for c in candidates
        if c.numbering and c.numbering.isdigit()
    ]
    if not numbered:
        return False

    sentence_like = sum(
        1 for candidate in numbered
        if candidate.title.endswith(".") or _SENTENCE_START_RE.match(candidate.title)
    )
    imperative_starts = {
        "add",
        "prepare",
        "incubate",
        "centrifuge",
        "remove",
        "measure",
        "mix",
        "place",
        "collect",
        "wash",
        "dilute",
        "pipette",
        "recall",
        "compare",
        "check",
    }
    imperative_like = sum(
        1 for candidate in numbered
        if candidate.title.split()
        and candidate.title.split()[0].lower().strip(".,;:") in imperative_starts
    )
    return (sentence_like + imperative_like) >= max(2, len(numbered) // 2)


def _candidate_line_gaps(candidates: List[_Candidate]) -> List[int]:
    ordered = sorted(candidates, key=lambda c: (c.source_block, c.source_line, c.source_order))
    gaps = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous.source_block == current.source_block:
            gaps.append(max(0, current.source_line - previous.source_line))
        else:
            gaps.append(999)
    return gaps


def _candidate_number_runs(candidates: List[_Candidate]) -> List[List[_Candidate]]:
    runs: List[List[_Candidate]] = []
    current: List[_Candidate] = []
    previous_number = None
    for candidate in candidates:
        try:
            number = int(candidate.numbering or "")
        except ValueError:
            continue
        if previous_number is None or number == previous_number + 1:
            current.append(candidate)
        else:
            if current:
                runs.append(current)
            current = [candidate]
        previous_number = number
    if current:
        runs.append(current)
    return runs


def _classify_sequence_run(
    run: List[_Candidate],
    region: SourceStructureRegion,
) -> Tuple[str, str, str, str, float]:
    if any(c.detection_method == "procedural_numbered_sentence" for c in run):
        return (
            "procedural_sequence",
            SOURCE_STRUCTURE_HIGH,
            "Numbered run contains sentence-like procedural steps",
            "procedural_sequence_support",
            -0.85,
        )

    gaps = _candidate_line_gaps(run)
    dense = bool(gaps and max(gaps) <= 1)

    if _numbered_run_is_predominantly_content_like(run):
        return (
            "local_enumeration",
            SOURCE_STRUCTURE_MEDIUM,
            "Numbered run is predominantly interrogative/prose-like content",
            "content_like_sequence_penalty",
            -0.6,
        )

    if region.region_type == "toc_like" or (
        region.region_type == "structural_outline"
        and run[0].hierarchy_level > 1
    ):
        return (
            "toc_sequence",
            SOURCE_STRUCTURE_HIGH,
            "Compact numbered run appears inside TOC/outline-like structural context",
            "toc_density_support",
            0.78,
        )

    if region.region_type == "semantic_local_scope":
        return (
            "semantic_local_sequence",
            SOURCE_STRUCTURE_HIGH,
            "Numbered run is locally structural under a semantic/textual parent",
            "semantic_local_scope_support",
            0.76,
        )

    if region.region_type == "local_enumeration" or dense:
        return (
            "local_enumeration",
            SOURCE_STRUCTURE_MEDIUM,
            "Dense numbered run lacks substantive separation and behaves like a local list",
            "local_enumeration_support",
            -0.55,
        )

    if gaps and max(gaps) >= 2:
        return (
            "structural_sequence",
            SOURCE_STRUCTURE_HIGH,
            "Numbered run has substantive content separation between entries",
            "substantive_content_separation_support",
            0.78,
        )

    return (
        "uncertain_sequence",
        SOURCE_STRUCTURE_LOW,
        "Numbered run lacks enough context to classify confidently",
        "region_ambiguity_penalty",
        -0.25,
    )


def _relationship_is_structurally_compatible(
    parent: _Candidate,
    child: _Candidate,
    candidates: Iterable[_Candidate],
    regions: Iterable[SourceStructureRegion],
    section_scopes: Iterable[SourceStructureScope],
) -> bool:
    if parent.status != STATUS_ACCEPTED or child.status != STATUS_ACCEPTED:
        return False

    parent_region = _region_for_candidate(parent, regions)
    child_region = _region_for_candidate(child, regions)

    if _explicit_numbering_relationship_is_safe(parent, child, candidates, parent_region, child_region):
        return True

    if (
        parent_region
        and child_region
        and parent_region.id != child_region.id
        and not (
            parent.detection_method == "structural_label"
            and child.detection_method == "structural_label"
        )
        and not _same_section_scope(parent, child, section_scopes)
    ):
        return False

    blocked_regions = {"local_enumeration", "procedural", "unknown"}
    if (
        parent_region
        and parent_region.region_type in blocked_regions
        and not _same_section_scope(parent, child, section_scopes)
    ):
        return False
    if child_region and child_region.region_type == "procedural":
        return False

    if parent.numbering and child.numbering and "." in child.numbering:
        return False

    if parent.detection_method == "structural_label":
        return child.hierarchy_level == parent.hierarchy_level + 1

    if parent.detection_method == "textual":
        return (
            (
                parent_region is not None
                and parent_region.region_type in {"mixed", "semantic_local_scope"}
            )
            or _same_section_scope(parent, child, section_scopes)
        ) and child.hierarchy_level == parent.hierarchy_level + 1

    return child.hierarchy_level == parent.hierarchy_level + 1


def _explicit_numbering_relationship_is_safe(
    parent: _Candidate,
    child: _Candidate,
    candidates: Iterable[_Candidate],
    parent_region: Optional[SourceStructureRegion],
    child_region: Optional[SourceStructureRegion],
) -> bool:
    if parent.status != STATUS_ACCEPTED or child.status != STATUS_ACCEPTED:
        return False
    if parent.source_order >= child.source_order:
        return False
    if not _is_exact_numbering_descendant(parent.numbering, child.numbering):
        return False
    if _direct_parent_numbering(child.numbering) != parent.numbering:
        return False
    if child.hierarchy_level != parent.hierarchy_level + 1:
        return False
    if not _is_safe_numbering(parent.numbering or ""):
        return False
    if not _is_safe_numbering(child.numbering or ""):
        return False
    if _looks_like_non_rehabilitatable_numbered_list_item(parent):
        return False
    if _looks_like_non_rehabilitatable_numbered_list_item(child):
        return False
    if (
        parent_region
        and child_region
        and parent_region.id != child_region.id
        and _accepted_direct_explicit_child_count(parent, candidates) < 2
    ):
        return False
    return True


def _accepted_direct_explicit_child_count(
    parent: _Candidate,
    candidates: Iterable[_Candidate],
) -> int:
    return sum(
        1
        for candidate in candidates
        if candidate.status == STATUS_ACCEPTED
        and _direct_parent_numbering(candidate.numbering) == parent.numbering
        and candidate.source_order > parent.source_order
        and _is_safe_numbering(candidate.numbering or "")
    )


def _apply_region_hierarchy_context(candidates: List[_Candidate], region_type: str) -> None:
    """Use local outline context to prevent compact outlines from flattening.

    Numbered entries in a table-of-contents or explicit outline region can be
    visually compact while still clearly nested under the nearest PART/CHAPTER/
    SECTION label. This helper adjusts only local diagnostic hierarchy levels;
    it does not create, remove, rename, or reorder candidates.
    """

    current_structural_level: Optional[int] = None
    for candidate in sorted(candidates, key=lambda c: c.source_order):
        if candidate.detection_method == "structural_label" or (
            region_type == "semantic_local_scope"
            and candidate.detection_method == "textual"
        ):
            current_structural_level = candidate.hierarchy_level
            continue

        if (
            current_structural_level is not None
            and candidate.numbering
            and candidate.numbering.isdigit()
            and candidate.hierarchy_level <= current_structural_level
        ):
            candidate.hierarchy_level = current_structural_level + 1


def _ordered_top_level_numbers(
    candidates: Iterable[_Candidate],
    contained_local_integer_ids: Optional[set] = None,
) -> List[int]:
    contained_local_integer_ids = contained_local_integer_ids or set()
    numbers = []
    for candidate in sorted(candidates, key=lambda c: c.source_order):
        if candidate.id in contained_local_integer_ids:
            continue
        if candidate.numbering and candidate.numbering.isdigit():
            value = int(candidate.numbering)
            if 1 <= value <= _MAX_TOP_LEVEL_SECTION:
                numbers.append(value)
    return numbers


def _ordered_roman_numbers(candidates: Iterable[_Candidate]) -> List[int]:
    numbers = []
    for candidate in sorted(candidates, key=lambda c: c.source_order):
        if candidate.detection_method in {"roman", "split_roman"} and candidate.numbering:
            value = _roman_to_int(candidate.numbering)
            if 1 <= value <= 30:
                numbers.append(value)
    return numbers


def _number_has_sequence_support(number: int, sequence: List[int]) -> bool:
    if sequence.count(number) != 1:
        return False
    sequence_set = set(sequence)
    neighbors = int(number - 1 in sequence_set) + int(number + 1 in sequence_set)
    if neighbors >= 1 and len(sequence) >= 3:
        return True
    sorted_unique = sorted(sequence_set)
    runs = _consecutive_runs(sorted_unique)
    return any(number in run and len(run) >= 3 for run in runs)


def _has_coherent_top_level_sequence(candidates: Iterable[_Candidate]) -> bool:
    return any(len(run) >= 3 for run in _consecutive_runs(_ordered_top_level_numbers(candidates)))


def _consecutive_runs(numbers: List[int]) -> List[List[int]]:
    if not numbers:
        return []
    runs = [[numbers[0]]]
    for number in numbers[1:]:
        if number == runs[-1][-1] + 1:
            runs[-1].append(number)
        elif number != runs[-1][-1]:
            runs.append([number])
    return runs


def _textual_heading_patterns(candidates: Iterable[_Candidate]) -> Dict[str, int]:
    patterns: Dict[str, int] = {}
    for candidate in candidates:
        if candidate.detection_method != "textual":
            continue
        key = _textual_pattern_key(candidate.title)
        patterns[key] = patterns.get(key, 0) + 1
    return patterns


def _textual_pattern_key(title: str) -> str:
    letters = [char for char in title if char.isalpha()]
    uppercase_ratio = (
        sum(1 for char in letters if char.isupper()) / len(letters)
        if letters else 0
    )
    word_count = len(title.split())
    if uppercase_ratio >= 0.75:
        return f"uppercase:{min(word_count, 4)}"
    return f"titlecase:{min(word_count, 4)}"


def _has_structural_label_context(candidates: Iterable[_Candidate]) -> bool:
    return any(c.detection_method == "structural_label" for c in candidates)


def _roman_to_int(value: str) -> int:
    total = 0
    previous = 0
    for char in reversed((value or "").upper()):
        current = _ROMAN_VALUES.get(char, 0)
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def _numbered_hierarchy_stats(headings: Iterable[SourceHeading]) -> dict:
    headings = list(headings)
    child_counts: Dict[str, int] = {}
    levels = set()
    for heading in headings:
        if not heading.numbering:
            continue
        levels.add(heading.hierarchy_level)
        parts = heading.numbering.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            child_counts[parent] = child_counts.get(parent, 0) + 1
    strongest_parent = None
    strongest_parent_child_count = 0
    for parent, count in sorted(child_counts.items()):
        if count > strongest_parent_child_count:
            strongest_parent = parent
            strongest_parent_child_count = count
    return {
        "depth": max(levels) if levels else 0,
        "strongest_parent": strongest_parent,
        "strongest_parent_child_count": strongest_parent_child_count,
    }


def _max_hierarchy_depth(candidates: Iterable[_Candidate]) -> int:
    levels = [candidate.hierarchy_level for candidate in candidates if candidate.status == STATUS_ACCEPTED]
    return max(levels) if levels else 0
