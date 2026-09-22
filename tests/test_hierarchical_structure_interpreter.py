import json
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import (
    ROLE_BODY,
    ROLE_HEADING,
    ROLE_LIST,
    ROLE_TITLE,
    CanonicalBlock,
    CanonicalDocumentInput,
    CanonicalUnit,
    SourcePosition,
)
from hierarchical_structure_interpreter import (
    GLOBAL_RECONNAISSANCE_SCHEMA,
    LOCAL_ANALYSIS_TASK,
    LOCAL_CLASS_LOCAL,
    LOCAL_CLASS_NON_STRUCTURAL,
    LOCAL_CLASS_STRUCTURAL,
    LOCAL_STRUCTURE_SCHEMA,
    MAX_HIERARCHICAL_MODEL_CALLS,
    STRUCTURAL_ROLE_CONTINUATION,
    STRUCTURAL_ROLE_LOCAL,
    STRUCTURAL_ROLE_SECTION,
    STRUCTURE_STATUS_WEAK,
    UniversalStructureError,
    build_document_input,
    compute_mapping_fidelity,
    discover_structural_candidates,
    fallback_windows,
    ground_macro_regions,
    interpret_hierarchical_structure,
    local_structure_from_json,
    prepare_candidate_analysis_scopes,
    prepare_local_analysis_scopes,
    propose_candidate_groups,
    reconnaissance_from_json,
    reconcile_macro_boundaries,
    reconcile_hierarchy,
    resolve_macro_region_blocks,
    strict_schema_violations,
    validate_local_candidate_sections,
)
from universal_structure_mapping import STATUS_ACCEPTED, STATUS_PARTIAL, build_mapping_result


def block(unit_type, unit_index, block_index, text, role=ROLE_BODY, style=None, hierarchy=None, order=None):
    order = order if order is not None else (unit_index * 100) + block_index
    return CanonicalBlock(
        block_id=f"{unit_type}-{unit_index}-{block_index}",
        text=text,
        raw_text=text,
        source_order=order,
        source_position=SourcePosition(unit_type, unit_index, block_index, 0, len(text)),
        role_hint=role,
        style_hint=style,
        native_hierarchy_hint=hierarchy,
    )


def doc(blocks, file_type="pdf", unit_type="page"):
    units_by_index = {}
    for item in blocks:
        units_by_index.setdefault(item.source_position.unit_index, []).append(item)
    units = [
        CanonicalUnit(
            unit_id=f"{unit_type}-{index}",
            unit_type=unit_type,
            unit_index=index,
            display_label=f"{unit_type.title()} {index}",
            blocks=unit_blocks,
        )
        for index, unit_blocks in sorted(units_by_index.items())
    ]
    return CanonicalDocumentInput(file_type, f"fixture.{file_type}", "Fixture", units)


def position(unit_type="page", unit=1, block_index=1, start=0, end=None):
    return {
        "unit_type": unit_type,
        "unit_index": unit,
        "block_index": block_index,
        "start_offset": start,
        "end_offset": end,
    }


def recon_json(regions, confidence="HIGH"):
    return {
        "document_title": "Fixture",
        "document_type": "notes",
        "structure_confidence": confidence,
        "organization_style": "hierarchical",
        "macro_regions": regions,
        "notes": [],
    }


DEFAULT_POSITION = object()
DEFAULT_SOURCE = object()


def region(title, source=DEFAULT_SOURCE, start=DEFAULT_POSITION, end=None, confidence="HIGH"):
    return {
        "semantic_title": title,
        "source_title": title if source is DEFAULT_SOURCE else source,
        "approximate_start": position() if start is DEFAULT_POSITION else start,
        "approximate_end": end,
        "confidence": confidence,
        "evidence_types": ["source_order", "native_metadata"],
    }


def local_json(sections, continuations=None, local_only=None, confidence="HIGH"):
    return {
        "region_title": "Region",
        "region_confidence": confidence,
        "sections": sections,
        "continuation_nodes": continuations or [],
        "local_only_nodes": local_only or [],
        "notes": [],
    }


def local_section(title, level=1, parent=None, role=STRUCTURAL_ROLE_SECTION, classification=LOCAL_CLASS_STRUCTURAL, start=None):
    return {
        "source_title": title,
        "semantic_title": title,
        "level": level,
        "parent": parent,
        "expected_start": start or position(),
        "expected_end": None,
        "confidence": "HIGH",
        "structural_role": role,
        "classification": classification,
    }


def compact_local_json(sections, confidence="HIGH"):
    return {
        "region_confidence": confidence,
        "sections": sections,
    }


def compact_boundary(title, level=1, parent=None, role=STRUCTURAL_ROLE_SECTION, start=None):
    return {
        "candidate_id": None,
        "group_id": None,
        "source_title": title,
        "level": level,
        "parent": parent,
        "expected_start": start or position(),
        "expected_end": None,
        "confidence": "HIGH",
        "structural_role": role,
    }


class FakeResponses:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            output_text = payload
        else:
            output_text = json.dumps(payload)
        return types.SimpleNamespace(
            output_text=output_text,
            status="completed",
            incomplete_details=None,
            max_output_tokens=6000,
            output=[types.SimpleNamespace(status="completed", content=[])],
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                output_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
            ),
        )


class FakeClient:
    def __init__(self, payloads):
        self.responses = FakeResponses(payloads)


class HierarchicalStructureInterpreterTests(unittest.TestCase):
    def test_01_clean_hierarchical_document(self):
        document = doc([block("page", 1, 1, "Part One\nA\nTopic A\nBody\nTopic B\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Part One")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("Topic A"),
            local_section("Topic B", start=position(start=21)),
        ]))
        canonical = reconcile_hierarchy(document, recon, [local])
        self.assertGreaterEqual(len(canonical.sections), 3)
        mapping = build_mapping_result(document, canonical.to_g2_document_structure())
        self.assertIn(mapping.mapping_status, {STATUS_ACCEPTED, STATUS_PARTIAL})

    def test_02_misleading_native_heading_style_is_evidence_not_truth(self):
        document = doc([
            block("document_flow", 1, 1, "Styled label", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Actual Section", role=ROLE_BODY),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([
            region("Actual Section", start=position("document_flow", 1, 2)),
        ]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("Actual Section", start=position("document_flow", 1, 2)),
        ]))
        canonical = reconcile_hierarchy(document, recon, [local])
        titles = [section.source_title for section in canonical.sections]
        self.assertNotIn("Styled label", titles)

    def test_03_semantic_section_without_native_heading_can_be_model_only(self):
        recon = reconnaissance_from_json(recon_json([region("Inferred Concept", source=None, start=None)]))
        local = local_structure_from_json("region-0001", local_json([
            {**local_section("Inferred Concept"), "source_title": None, "expected_start": None},
        ]))
        canonical = reconcile_hierarchy(doc([block("page", 1, 1, "Body")]), recon, [local])
        self.assertGreater(canonical.diagnostics["unresolved_model_only_sections"], 0)

    def test_04_local_list_not_promoted_globally(self):
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json(
            [],
            local_only=[
                {"title": "1 Local item", "kind": "enumeration", "expected_start": position(), "confidence": "HIGH"}
            ],
        ))
        canonical = reconcile_hierarchy(doc([block("page", 1, 1, "Main\n1 Local item")]), recon, [local])
        self.assertEqual(canonical.diagnostics["local_only_nodes"], 1)
        self.assertEqual(len([s for s in canonical.sections if s.source_title == "1 Local item"]), 0)

    def test_05_repeated_heading_as_continuation(self):
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json(
            [local_section("Topic")],
            continuations=[{"title": "Topic", "expected_start": position(start=20), "confidence": "HIGH"}],
        ))
        canonical = reconcile_hierarchy(doc([block("page", 1, 1, "Main\nTopic\nBody\nTopic")]), recon, [local])
        self.assertEqual(canonical.diagnostics["continuation_nodes"], 1)

    def test_06_diagram_like_labels_remain_local(self):
        recon = reconnaissance_from_json(recon_json([region("Diagram")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("Label A", role=STRUCTURAL_ROLE_LOCAL, classification=LOCAL_CLASS_LOCAL),
            local_section("Label B", role=STRUCTURAL_ROLE_LOCAL, classification=LOCAL_CLASS_NON_STRUCTURAL),
        ]))
        canonical = reconcile_hierarchy(doc([block("slide", 1, 1, "Diagram\nLabel A\nLabel B")], file_type="pptx", unit_type="slide"), recon, [local])
        self.assertEqual(canonical.diagnostics["local_only_nodes"], 2)

    def test_07_multiple_macro_regions(self):
        recon = reconnaissance_from_json(recon_json([
            region("One", start=position(unit=1)),
            region("Two", start=position(unit=2)),
        ]))
        self.assertEqual(len(recon.macro_regions), 2)

    def test_08_nested_local_hierarchy(self):
        document = doc([block("page", 1, 1, "Main\nParent\nChild\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("Parent"),
            local_section("Child", level=2, parent="Parent", start=position(start=12)),
        ]))
        canonical = reconcile_hierarchy(document, recon, [local])
        child = next(section for section in canonical.sections if section.source_title == "Child")
        self.assertIsNotNone(child.parent_id)

    def test_09_duplicate_titles_in_different_regions_are_allowed(self):
        recon = reconnaissance_from_json(recon_json([
            region("One"),
            region("Two", start=position(unit=2)),
        ]))
        local_a = local_structure_from_json("region-0001", local_json([local_section("Methods")]))
        local_b = local_structure_from_json("region-0002", local_json([local_section("Methods", start=position(unit=2))]))
        canonical = reconcile_hierarchy(doc([
            block("page", 1, 1, "One\nMethods"),
            block("page", 2, 1, "Two\nMethods"),
        ]), recon, [local_a, local_b])
        self.assertEqual(len([s for s in canonical.sections if s.source_title == "Methods"]), 2)

    def test_10_repeated_title_at_distinct_positions_is_preserved(self):
        recon = reconnaissance_from_json(recon_json([region("One")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("Repeat"),
            local_section("Repeat", start=position(start=20)),
        ]))
        canonical = reconcile_hierarchy(doc([block("page", 1, 1, "One\nRepeat\nRepeat")]), recon, [local])
        self.assertEqual(canonical.diagnostics["duplicate_canonical_paths"], 0)
        self.assertEqual(len([s for s in canonical.sections if s.source_title == "Repeat"]), 2)

    def test_11_unresolved_source_anchor(self):
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([local_section("Missing")]))
        canonical = reconcile_hierarchy(doc([block("page", 1, 1, "Main only")]), recon, [local])
        mapping = build_mapping_result(doc([block("page", 1, 1, "Main only")]), canonical.to_g2_document_structure())
        self.assertGreater(mapping.unresolved_sections, 0)

    def test_12_source_order_conflict(self):
        document = doc([block("page", 1, 1, "Second\nFirst")])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("First", start=position(start=7)),
            local_section("Second", start=position(start=0)),
        ]))
        mapping = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        self.assertGreater(mapping.source_order_violations, 0)

    def test_13_overlapping_macro_region_boundary_detected(self):
        recon = reconnaissance_from_json(recon_json([
            region("One", start=position(unit=1), end=position(unit=3)),
            region("Two", start=position(unit=2), end=position(unit=4)),
        ]))
        resolved = resolve_macro_region_blocks(doc([
            block("page", 1, 1, "One"),
            block("page", 2, 1, "Two"),
            block("page", 3, 1, "Three"),
            block("page", 4, 1, "Four"),
        ]), recon)
        resolved_recon = recon_json([])
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved_recon["macro_regions"], [])

    def test_14_content_before_first_heading_is_preserved_in_segments(self):
        document = doc([block("page", 1, 1, "Preface text\nMain\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Main", start=position(start=13))]))
        local = local_structure_from_json("region-0001", local_json([local_section("Main", start=position(start=13))]))
        result = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        self.assertEqual(result.segments[0].section_path, "Fixture")

    def test_15_same_unit_multiple_headings(self):
        document = doc([block("page", 1, 1, "Main\nA\nBody\nB\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("A", start=position(start=5)),
            local_section("B", start=position(start=12)),
        ]))
        result = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        self.assertTrue(any(segment.section_path.endswith("A") for segment in result.segments))
        self.assertTrue(any(segment.section_path.endswith("B") for segment in result.segments))

    def test_16_region_continuation_across_units(self):
        document = doc([block("page", 1, 1, "Main\nBody"), block("page", 2, 1, "Continuation")])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([local_section("Main")]))
        result = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        self.assertEqual(result.segments[-1].section_path.split(" > ")[0], "Main")

    def test_17_large_synthetic_document_uses_multiple_fallback_windows(self):
        document = doc([block("page", i, 1, f"Block {i}") for i in range(1, 190)])
        windows = fallback_windows(document)
        self.assertGreater(len(windows), 1)
        self.assertLess(windows[0][1][-1].source_order, windows[1][1][-1].source_order)

    def test_18_fallback_windowing_when_reconnaissance_has_no_boundaries(self):
        document = doc([block("page", i, 1, f"Block {i}") for i in range(1, 100)])
        recon = reconnaissance_from_json(recon_json([], confidence="LOW"))
        self.assertGreater(len(resolve_macro_region_blocks(document, recon)), 1)

    def test_19_same_title_distinct_source_offsets_are_not_title_deduplicated(self):
        recon = reconnaissance_from_json(recon_json([region("Only")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("A"),
            local_section("A", start=position(start=20)),
        ]))
        canonical = reconcile_hierarchy(doc([block("page", 1, 1, "Only\nA\nA")]), recon, [local])
        self.assertEqual(len([s for s in canonical.sections if s.source_title == "A"]), 2)

    def test_20_truncated_or_failed_local_model_call_is_contained(self):
        document = doc([block("page", 1, 1, "Main")])
        client = FakeClient([
            recon_json([region("Main")]),
            "{",
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertTrue(result.local_structures[0].interpretation_failed)
        self.assertEqual(result.canonical_structure.structure_quality.status, STRUCTURE_STATUS_WEAK)

    def test_21_partial_safe_mapping(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("A", start=position(start=5)),
            local_section("Missing", start=position(unit=2)),
        ]))
        result = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        self.assertEqual(result.mapping_status, STATUS_PARTIAL)

    def test_22_high_mapping_fidelity_low_structure_quality_kept_separate(self):
        document = doc([block("page", 1, 1, "A")])
        recon = reconnaissance_from_json(recon_json([region("A")], confidence="LOW"))
        local = local_structure_from_json("region-0001", local_json([local_section("A")], confidence="LOW"))
        canonical = reconcile_hierarchy(document, recon, [local])
        mapping = build_mapping_result(document, canonical.to_g2_document_structure())
        fidelity = compute_mapping_fidelity(mapping)
        self.assertNotEqual(canonical.structure_quality.score, fidelity.score)

    def test_23_high_structure_quality_partial_mapping_fidelity(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("A", start=position(start=5)),
            local_section("Missing", start=position(unit=2)),
        ]))
        canonical = reconcile_hierarchy(document, recon, [local])
        self.assertGreaterEqual(canonical.structure_quality.score, 0.5)
        self.assertEqual(compute_mapping_fidelity(build_mapping_result(document, canonical.to_g2_document_structure())).status, "partial")

    def test_24_no_chunk_crossing_canonical_boundary(self):
        raw = "Main\nA\n" + ("Alpha " * 120) + "\nB\n" + ("Beta " * 120)
        document = doc([block("page", 1, 1, raw)])
        recon = reconnaissance_from_json(recon_json([region("Main")]))
        local = local_structure_from_json("region-0001", local_json([
            local_section("A", start=position(start=5)),
            local_section("B", start=position(start=730)),
        ]))
        result = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        for chunk in result.diagnostic_chunks:
            self.assertFalse("Alpha" in chunk.text and "Beta" in chunk.text)

    def test_25_strict_json_schema_validity_recursively(self):
        self.assertEqual(strict_schema_violations(GLOBAL_RECONNAISSANCE_SCHEMA), [])
        self.assertEqual(strict_schema_violations(LOCAL_STRUCTURE_SCHEMA), [])

    def test_pdf_global_recon_uses_native_input_file(self):
        document = doc([block("page", 1, 1, "Main")])
        payload = build_document_input("Task", document, original_file_bytes=b"%PDF")
        content = payload[0]["content"]
        self.assertEqual(content[1]["type"], "input_file")
        self.assertIn("data:application/pdf;base64,", content[1]["file_data"])

    def test_docx_local_analysis_uses_structured_native_evidence(self):
        document = doc(
            [block("document_flow", 1, 1, "Heading", role=ROLE_HEADING, style="Heading 1", hierarchy=1)],
            file_type="docx",
            unit_type="document_flow",
        )
        payload = build_document_input(LOCAL_ANALYSIS_TASK, document)
        text = payload[0]["content"][1]["text"]
        self.assertIn("role_hint=heading", text)
        self.assertIn("hierarchy_hint=1", text)

    def test_pptx_local_analysis_uses_slide_evidence(self):
        document = doc([block("slide", 1, 1, "Slide Title", role=ROLE_TITLE)], file_type="pptx", unit_type="slide")
        payload = build_document_input(LOCAL_ANALYSIS_TASK, document)
        self.assertIn("[unit=slide:1]", payload[0]["content"][1]["text"])

    def test_model_call_pipeline_reports_call_count(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        client = FakeClient([
            recon_json([region("Main")]),
            local_json([local_section("A", start=position(start=5))]),
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertEqual(result.diagnostics["model_calls"], 2)
        self.assertEqual(len(client.responses.calls), 2)

    def test_g31_null_positions_can_ground_by_exact_source_title(self):
        document = doc([block("page", 1, 1, "Literal Heading\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Semantic", source="Literal Heading", start=None)]))
        grounding = ground_macro_regions(document, recon)[0]
        self.assertEqual(grounding.method, "exact_source_title")
        self.assertEqual(grounding.block_list_index, 0)

    def test_g31_whitespace_normalized_source_title_grounding(self):
        document = doc([block("page", 1, 1, "Literal   Heading\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Semantic", source="Literal Heading", start=None)]))
        self.assertEqual(ground_macro_regions(document, recon)[0].method, "whitespace_source_title")

    def test_g31_unicode_normalized_source_title_grounding(self):
        document = doc([block("page", 1, 1, "Café Biology\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Semantic", source="Café Biology", start=None)]))
        self.assertEqual(ground_macro_regions(document, recon)[0].method, "unicode_source_title")

    def test_g31_punctuation_normalized_source_title_grounding(self):
        document = doc([block("page", 1, 1, "Gene: regulation\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Semantic", source="Gene regulation", start=None)]))
        self.assertEqual(ground_macro_regions(document, recon)[0].method, "punctuation_source_title")

    def test_g31_conservative_fuzzy_source_title_grounding(self):
        document = doc([block("page", 1, 1, "Genetic variation and inheritance\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Semantic", source="Genetic variations and inheritance", start=None)]))
        self.assertEqual(ground_macro_regions(document, recon)[0].method, "fuzzy_source_title")

    def test_g31_semantic_title_not_used_as_source_anchor(self):
        document = doc([block("page", 1, 1, "Literal Heading\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Literal Heading", source=None, start=None)]))
        grounding = ground_macro_regions(document, recon)[0]
        self.assertEqual(grounding.status, "unresolved")

    def test_g31_unresolved_regions_do_not_become_full_document_scopes(self):
        document = doc([block("page", i, 1, f"Page {i}") for i in range(1, 50)])
        recon = reconnaissance_from_json(recon_json([
            region("A", source=None, start=None),
            region("B", source=None, start=None),
            region("C", source=None, start=None),
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertTrue(all(scope.scope_type == "fallback_window" for scope in scopes))
        self.assertFalse(any(scope.block_count == len(document.blocks) for scope in scopes))

    def test_g31_partially_grounded_unresolved_region_is_skipped_not_full_document(self):
        document = doc([
            block("page", 1, 1, "A\nBody"),
            block("page", 2, 1, "Body"),
            block("page", 3, 1, "C\nBody"),
        ])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A", start=None),
            region("B", source=None, start=None),
            region("C", source="C", start=None),
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        unresolved = [scope for scope in scopes if scope.scope_type == "unresolved_region"]
        self.assertEqual(len(unresolved), 1)
        self.assertTrue(unresolved[0].skipped)
        self.assertEqual(unresolved[0].block_count, 0)

    def test_g31_next_grounded_heading_defines_previous_region_end(self):
        document = doc([
            block("page", 1, 1, "A\nBody"),
            block("page", 2, 1, "Middle"),
            block("page", 3, 1, "C\nBody"),
        ])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A", start=None),
            region("C", source="C", start=None),
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(scopes[0].block_count, 2)

    def test_g31_duplicate_heading_collision_unresolves_second_region(self):
        document = doc([block("page", 1, 1, "Repeat\nBody")])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="Repeat", start=None),
            region("B", source="Repeat", start=None),
        ]))
        groundings = ground_macro_regions(document, recon)
        self.assertEqual(groundings[0].status, "grounded")
        self.assertEqual(groundings[1].status, "unresolved")

    def test_g31_source_order_violation_unresolves_late_region(self):
        document = doc([block("page", 1, 1, "Second\nFirst")])
        recon = reconnaissance_from_json(recon_json([
            region("First", source="First", start=None),
            region("Second", source="Second", start=None),
        ]))
        groundings = ground_macro_regions(document, recon)
        self.assertEqual(groundings[1].method, "source_order_rejected")

    def test_g31_fallback_windows_are_bounded(self):
        document = doc([block("page", i, 1, f"Page {i}") for i in range(1, 100)])
        recon = reconnaissance_from_json(recon_json([], confidence="LOW"))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertGreater(len(scopes), 1)
        self.assertTrue(all(scope.block_count <= 80 for scope in scopes))

    def test_g31_fallback_windows_do_not_become_canonical_headings(self):
        document = doc([block("page", i, 1, f"Page {i}") for i in range(1, 100)])
        recon = reconnaissance_from_json(recon_json([], confidence="LOW"))
        scopes = prepare_local_analysis_scopes(document, recon, [])
        locals_ = [
            local_structure_from_json(scope.region.region_id, local_json([local_section("A")]))
            for scope in scopes[:1]
        ]
        canonical = reconcile_hierarchy(document, recon, locals_)
        self.assertEqual(canonical.sections, [])

    def test_g31_large_whole_document_grounded_region_is_skipped(self):
        document = doc([block("page", i, 1, "A" if i == 1 else f"Page {i}") for i in range(1, 50)])
        recon = reconnaissance_from_json(recon_json([region("A", source="A", start=None)]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertGreater(len(scopes), 1)
        self.assertFalse(any(scope.block_count == len(document.blocks) for scope in scopes))
        self.assertTrue(all(scope.subdivision_reason == "local_scope_budget_exceeded" for scope in scopes))

    def test_g31_small_document_whole_document_case_is_explicit(self):
        document = doc([block("page", 1, 1, "Small"), block("page", 2, 1, "Doc")])
        recon = reconnaissance_from_json(recon_json([region("Unknown", source=None, start=None)]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(scopes[0].scope_type, "small_document")
        self.assertFalse(scopes[0].skipped)

    def test_g31_local_parse_failure_preserves_error_reason(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            "not-json",
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertIn("valid structured JSON", result.local_structures[0].diagnostics["failure_reason"])

    def test_g31_incomplete_max_output_diagnostics_are_preserved(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            "{",
        ])
        response = client.responses.create
        def incomplete_response(**kwargs):
            output_text = "{"
            client.responses.calls.append(kwargs)
            return types.SimpleNamespace(
                output_text=output_text,
                status="incomplete",
                incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
                max_output_tokens=6000,
                output=[types.SimpleNamespace(status="incomplete", content=[])],
                usage=types.SimpleNamespace(
                    input_tokens=11,
                    output_tokens=6000,
                    total_tokens=6011,
                    output_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
                ),
            )
        client.responses.create = lambda **kwargs: response(**kwargs) if len(client.responses.calls) == 0 else incomplete_response(**kwargs)
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        diagnostics = result.local_structures[0].diagnostics
        self.assertEqual(diagnostics["response_status"], "incomplete")
        self.assertEqual(diagnostics["incomplete_reason"], "max_output_tokens")

    def test_g31_usage_aggregation_inputs_available_per_local_call(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            local_json([local_section("A", start=position(start=5))]),
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertEqual(result.local_structures[0].diagnostics["input_tokens"], 10)

    def test_g31_final_structure_remains_g2_compatible(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json("region-0001", local_json([local_section("A", start=position(start=5))]))
        g2_structure = reconcile_hierarchy(document, recon, [local]).to_g2_document_structure()
        self.assertTrue(hasattr(g2_structure, "sections"))
        self.assertGreaterEqual(len(g2_structure.sections), 1)

    def test_g32_large_grounded_region_subdivides_by_budget(self):
        blocks = [block("page", i, 1, ("A\n" if i == 1 else "") + ("x" * 3000)) for i in range(1, 11)]
        document = doc(blocks)
        recon = reconnaissance_from_json(recon_json([region("A", source="A", start=None)]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertGreater(len(scopes), 1)
        self.assertTrue(all(scope.input_char_count <= 16000 for scope in scopes))
        self.assertTrue(all(scope.subdivision_reason == "local_scope_budget_exceeded" for scope in scopes))

    def test_g32_small_grounded_region_remains_single_scope(self):
        document = doc([block("page", 1, 1, "A\nshort body")])
        recon = reconnaissance_from_json(recon_json([region("A", source="A", start=None)]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].scope_type, "grounded_region")
        self.assertEqual(scopes[0].subdivision_reason, "fits_local_budget")

    def test_g32_subscopes_are_monotonic_and_non_overlapping(self):
        blocks = [block("page", i, 1, ("A\n" if i == 1 else "") + ("x" * 3000)) for i in range(1, 11)]
        document = doc(blocks)
        recon = reconnaissance_from_json(recon_json([region("A", source="A", start=None)]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        previous_end = None
        for scope in scopes:
            self.assertIsNotNone(scope.source_start)
            self.assertIsNotNone(scope.source_end)
            if previous_end is not None:
                self.assertLess(previous_end, scope.source_start)
            previous_end = scope.source_end

    def test_g32_subscopes_cover_region_without_gaps(self):
        raw_a = "A\n" + ("alpha " * 500)
        raw_b = "beta " * 500
        raw_c = "C\n" + ("gamma " * 50)
        document = doc([
            block("page", 1, 1, raw_a),
            block("page", 2, 1, raw_b),
            block("page", 3, 1, raw_c),
        ])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A", start=None),
            region("C", source="C", start=None),
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        owned_a = "".join(block.raw_text for block in scopes[0].blocks)
        self.assertEqual(owned_a, raw_a + raw_b)

    def test_g32_same_block_macro_boundaries_do_not_overlap(self):
        raw = "A heading\n" + ("a" * 30) + "\nB heading\n" + ("b" * 30)
        document = doc([block("page", 1, 1, raw)])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A heading", start=None),
            region("B", source="B heading", start=None),
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(len([scope for scope in scopes if not scope.skipped]), 2)
        first, second = [scope for scope in scopes if not scope.skipped]
        self.assertIn("A heading", first.blocks[0].raw_text)
        self.assertNotIn("B heading", first.blocks[0].raw_text)
        self.assertIn("B heading", second.blocks[0].raw_text)
        self.assertNotIn("A heading", second.blocks[0].raw_text)

    def test_g32_one_block_slice_respects_start_and_end_offsets(self):
        raw = "prefix A start middle B end suffix"
        document = doc([block("page", 1, 1, raw)])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A start", start=None),
            region("B", source="B end", start=None),
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        first = [scope for scope in scopes if scope.macro_region_id == "region-0001"][0]
        self.assertEqual(first.blocks[0].raw_text, "A start middle ")
        self.assertEqual(first.source_start.start_offset, raw.index("A start"))
        self.assertEqual(first.source_end.end_offset, raw.index("B end"))

    def test_g32_adjacent_scope_text_is_not_duplicated(self):
        raw = "A\n" + ("a" * 100) + "\nB\n" + ("b" * 100)
        document = doc([block("page", 1, 1, raw)])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A", start=None),
            region("B", source="B", start=None),
        ]))
        scopes = [scope for scope in prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon)) if not scope.skipped]
        combined = "".join(block.raw_text for scope in scopes for block in scope.blocks)
        self.assertEqual(combined, raw)

    def test_g32_duplicate_continuation_across_subscopes_reconciles_once(self):
        document = doc([block("page", 1, 1, "Main\nRepeated")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local_a = local_structure_from_json("region-0001", local_json([local_section("Repeated")]))
        local_b = local_structure_from_json("region-0001", local_json([local_section("Repeated")]))
        local_b = type(local_b)(
            **{
                field: getattr(local_b, field)
                for field in type(local_b).__dataclass_fields__
                if field != "diagnostics"
            },
            diagnostics={"subscope_index": 2},
        )
        canonical = reconcile_hierarchy(document, recon, [local_a, local_b])
        self.assertEqual(len([section for section in canonical.sections if section.source_title == "Repeated"]), 1)
        self.assertEqual(canonical.diagnostics["duplicate_canonical_paths"], 1)

    def test_g32_one_failed_subscope_does_not_discard_successful_sibling(self):
        blocks = [block("page", 1, i, f"Heading {i}", role=ROLE_HEADING) for i in range(1, 120)]
        document = doc(blocks)
        client = FakeClient([
            recon_json([region("Heading 1", source="Heading 1", start=None)]),
            compact_local_json([compact_boundary("Heading 1", start=position(block_index=1))]),
            "{",
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertTrue(any(local.sections for local in result.local_structures))
        self.assertTrue(any(local.interpretation_failed for local in result.local_structures))
        self.assertTrue(any(section.source_title == "Heading 1" for section in result.canonical_structure.sections))

    def test_g32_many_unresolved_regions_do_not_create_n_full_document_calls(self):
        document = doc([block("page", i, 1, f"Page {i}") for i in range(1, 50)])
        recon = reconnaissance_from_json(recon_json([
            region(f"Semantic {i}", source=None, start=None) for i in range(1, 15)
        ]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertFalse(any(scope.block_count == len(document.blocks) for scope in scopes))
        self.assertLess(len(scopes), len(recon.macro_regions))

    def test_g32_scope_diagnostics_include_subdivision_and_offsets(self):
        document = doc([block("page", 1, 1, "Main\nA")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        scope = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))[0]
        diagnostics = scope and {
            "source_start": scope.source_start,
            "source_end": scope.source_end,
            "subdivision_reason": scope.subdivision_reason,
            "macro_region_id": scope.macro_region_id,
        }
        self.assertIsNotNone(diagnostics["source_start"])
        self.assertIsNotNone(diagnostics["source_end"])
        self.assertEqual(diagnostics["macro_region_id"], "region-0001")

    def test_g32_macro_region_count_can_differ_from_local_call_count(self):
        blocks = [block("page", i, 1, ("Main\n" if i == 1 else "") + ("x" * 3000)) for i in range(1, 10)]
        document = doc(blocks)
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        scopes = prepare_local_analysis_scopes(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(len(recon.macro_regions), 1)
        self.assertGreater(len(scopes), 1)

    def test_g33_compact_local_schema_has_no_verbose_fields(self):
        schema_text = json.dumps(LOCAL_STRUCTURE_SCHEMA)
        for forbidden in [
            "semantic_title",
            "classification",
            "continuation_nodes",
            "local_only_nodes",
            "notes",
            "reason",
            "description",
            "summary",
        ]:
            self.assertNotIn(forbidden, schema_text)
        section_props = LOCAL_STRUCTURE_SCHEMA["properties"]["sections"]["items"]["properties"]
        self.assertEqual(
            set(section_props),
            {
                "candidate_id",
                "group_id",
                "source_title",
                "level",
                "parent",
                "expected_start",
                "expected_end",
                "confidence",
                "structural_role",
            },
        )

    def test_g33_strict_compact_schema_validity_recursively(self):
        self.assertEqual(strict_schema_violations(LOCAL_STRUCTURE_SCHEMA), [])

    def test_g33_compact_local_preserves_literal_source_headings(self):
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([compact_boundary("Exact Literal Heading")]),
        )
        self.assertEqual(local.sections[0].source_title, "Exact Literal Heading")
        self.assertEqual(local.sections[0].semantic_title, "Exact Literal Heading")
        self.assertEqual(local.sections[0].classification, LOCAL_CLASS_STRUCTURAL)

    def test_g33_semantic_paraphrase_is_not_required_from_local_analysis(self):
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([compact_boundary("5.4 Trp operon: repressible operon")]),
        )
        self.assertEqual(local.sections[0].semantic_title, "5.4 Trp operon: repressible operon")

    def test_g33_bullets_are_not_promoted_when_marked_local(self):
        document = doc([block("page", 1, 1, "Main\n- Local bullet", role=ROLE_LIST)])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([
                compact_boundary("- Local bullet", role=STRUCTURAL_ROLE_LOCAL, start=position(start=5))
            ]),
        )
        canonical = reconcile_hierarchy(document, recon, [local])
        self.assertEqual(len([s for s in canonical.sections if s.source_title == "- Local bullet"]), 0)
        self.assertEqual(canonical.diagnostics["local_only_nodes"], 1)

    def test_g33_figure_table_labels_do_not_become_headings_when_local(self):
        document = doc([block("page", 1, 1, "Main\nFigure 1\nTable A")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([
                compact_boundary("Figure 1", role=STRUCTURAL_ROLE_LOCAL, start=position(start=5)),
                compact_boundary("Table A", role=STRUCTURAL_ROLE_LOCAL, start=position(start=14)),
            ]),
        )
        canonical = reconcile_hierarchy(document, recon, [local])
        self.assertEqual(canonical.diagnostics["local_only_nodes"], 2)
        self.assertNotIn("Figure 1", [section.source_title for section in canonical.sections])

    def test_g33_nested_compact_headings_produce_canonical_paths(self):
        document = doc([block("page", 1, 1, "Main\nParent\nChild")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([
                compact_boundary("Parent", start=position(start=5)),
                compact_boundary("Child", level=2, parent="Parent", start=position(start=12)),
            ]),
        )
        canonical = reconcile_hierarchy(document, recon, [local])
        child = next(section for section in canonical.sections if section.source_title == "Child")
        parent = next(section for section in canonical.sections if section.section_id == child.parent_id)
        self.assertEqual(parent.source_title, "Parent")

    def test_g33_two_compact_headings_in_one_block_keep_distinct_offsets(self):
        raw = "Main\nAlpha\nBody\nBeta\nBody"
        document = doc([block("page", 1, 1, raw)])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([
                compact_boundary("Alpha", start=position(start=raw.index("Alpha"))),
                compact_boundary("Beta", start=position(start=raw.index("Beta"))),
            ]),
        )
        canonical = reconcile_hierarchy(document, recon, [local])
        starts = {
            section.source_title: section.expected_start.start_offset
            for section in canonical.sections
            if section.source_title in {"Alpha", "Beta"}
        }
        self.assertLess(starts["Alpha"], starts["Beta"])

    def test_g33_unresolved_local_source_title_is_not_fabricated_into_heading(self):
        document = doc([block("page", 1, 1, "Main\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([compact_boundary(None, start=None)]),
        )
        canonical = reconcile_hierarchy(document, recon, [local])
        self.assertEqual(len([section for section in canonical.sections if section.level > 1]), 0)
        self.assertEqual(canonical.diagnostics["unresolved_model_only_sections"], 1)

    def test_g33_compact_result_remains_g2_compatible_and_literal_anchorable(self):
        document = doc([block("page", 1, 1, "Main\nLiteral Section\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([compact_boundary("Literal Section", start=position(start=5))]),
        )
        g2_structure = reconcile_hierarchy(document, recon, [local]).to_g2_document_structure()
        result = build_mapping_result(document, g2_structure)
        self.assertEqual(result.sections_anchored, len(g2_structure.sections))
        self.assertTrue(any(section.source_title == "Literal Section" for section in g2_structure.sections))

    def test_g33_no_diagnostic_chunk_crosses_canonical_section_boundary(self):
        document = doc([block("page", 1, 1, "Main\nA\nBody A\nB\nBody B")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        local = local_structure_from_json(
            "region-0001",
            compact_local_json([
                compact_boundary("A", start=position(start=5)),
                compact_boundary("B", start=position(start=14)),
            ]),
        )
        result = build_mapping_result(document, reconcile_hierarchy(document, recon, [local]).to_g2_document_structure())
        self.assertEqual(result.diagnostics.get("chunks_crossing_boundaries", 0), 0)

    def test_g33_successful_local_diagnostics_count_structural_boundaries(self):
        document = doc([block("page", 1, 1, "Main\nA\nB")])
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            compact_local_json([
                compact_boundary("A", start=position(start=5)),
                compact_boundary("B", start=position(start=7)),
            ]),
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertEqual(result.local_structures[0].diagnostics["structural_boundaries"], 2)

    def test_g33_compact_payload_is_smaller_than_previous_local_contract(self):
        compact = compact_local_json([compact_boundary(f"Heading {i}") for i in range(20)])
        verbose = local_json([local_section(f"Heading {i}") for i in range(20)])
        self.assertLess(len(json.dumps(compact)), len(json.dumps(verbose)))

    def test_g34_docx_native_heading_hints_produce_candidates(self):
        document = doc([
            block("document_flow", 1, 1, "Cardiac Muscle", role=ROLE_HEADING, style="Heading 1", hierarchy=1)
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Cardiac Muscle", source="Cardiac Muscle", start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertTrue(any("native_role" in ",".join(candidate.evidence_signals) for candidate in candidates))

    def test_g34_pptx_title_role_hints_produce_candidates(self):
        document = doc([
            block("slide", 1, 1, "Respiration", role=ROLE_TITLE)
        ], file_type="pptx", unit_type="slide")
        recon = reconnaissance_from_json(recon_json([region("Respiration", source="Respiration", start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertTrue(any(candidate.role_hint == ROLE_TITLE for candidate in candidates))

    def test_g34_pdf_heading_like_text_produces_candidate(self):
        document = doc([block("page", 1, 1, "5.4 Trp operon")])
        recon = reconnaissance_from_json(recon_json([region("Trp", source="5.4 Trp operon", start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertTrue(any(candidate.numbering_pattern == "dotted" for candidate in candidates))

    def test_g34_long_paragraph_is_not_candidate_by_itself(self):
        document = doc([block("page", 1, 1, "This is a long paragraph. " * 40)])
        recon = reconnaissance_from_json(recon_json([region("Semantic", source=None, start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(candidates, [])

    def test_g34_numbered_headings_can_become_candidates(self):
        document = doc([block("page", 1, 1, "12 Translation elongation")])
        recon = reconnaissance_from_json(recon_json([region("Translation", source="12 Translation elongation", start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertTrue(any(candidate.numbering_pattern == "integer" for candidate in candidates))

    def test_g34_global_macro_source_title_match_is_strong_candidate(self):
        document = doc([block("page", 1, 1, "Macro Title\nBody")])
        recon = reconnaissance_from_json(recon_json([region("Macro", source="Macro Title", start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertTrue(any("global_macro_source_title" in candidate.evidence_signals for candidate in candidates))

    def test_g34_candidate_positions_preserve_provenance_and_order(self):
        document = doc([
            block("page", 1, 1, "A", role=ROLE_HEADING),
            block("page", 2, 1, "B", role=ROLE_HEADING),
        ])
        recon = reconnaissance_from_json(recon_json([
            region("A", source="A", start=None),
            region("B", source="B", start=None),
        ]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertEqual([candidate.source_position.unit_index for candidate in candidates], [1, 2])
        self.assertLess(candidates[0].source_position, candidates[1].source_position)

    def test_g34_duplicate_candidate_evidence_deduplicates(self):
        document = doc([block("page", 1, 1, "A", role=ROLE_HEADING)])
        recon = reconnaissance_from_json(recon_json([region("A", source="A", start=None)]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        self.assertEqual(len(candidates), 1)

    def test_g34_model_can_reject_false_candidate(self):
        document = doc([block("page", 1, 1, "Main\nFigure 1")])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            compact_local_json([]),
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertFalse(any(section.source_title == "Figure 1" for section in result.canonical_structure.sections))

    def test_g34_model_can_accept_true_candidate_literal(self):
        document = doc([block("page", 1, 1, "Main", role=ROLE_HEADING)])
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            compact_local_json([compact_boundary("Main", start=position())]),
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertTrue(any(section.source_title == "Main" for section in result.canonical_structure.sections))

    def test_g34_candidate_batching_does_not_depend_on_raw_block_count(self):
        blocks = [block("document_flow", 1, i, f"Paragraph {i} body text.") for i in range(1, 2501)]
        for i in range(1, 51):
            blocks[i * 40 - 1] = block("document_flow", 1, i * 40, f"Heading {i}", role=ROLE_HEADING)
        document = doc(blocks, file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Heading 1", source="Heading 1", start=None)]))
        client = FakeClient([
            recon_json([region("Heading 1", source="Heading 1", start=None)]),
            compact_local_json([]),
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        calls = len(client.responses.calls)
        self.assertLess(calls, 10)
        self.assertLess(result.diagnostics["candidate_diagnostics"]["candidate_analysis_calls"], 10)

    def test_g34_model_call_budget_prevents_runaway_and_preserves_successes(self):
        blocks = [block("page", 1, i, f"Heading {i}", role=ROLE_HEADING) for i in range(1, 3000)]
        document = doc(blocks)
        client = FakeClient([
            recon_json([region("Heading 1", source="Heading 1", start=None)]),
            *[
                compact_local_json([compact_boundary("Heading 1", start=position(block_index=1))])
                for _ in range(MAX_HIERARCHICAL_MODEL_CALLS - 1)
            ],
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertLessEqual(len(client.responses.calls), MAX_HIERARCHICAL_MODEL_CALLS)
        self.assertTrue(result.diagnostics["candidate_diagnostics"]["call_budget_exhausted"])
        self.assertTrue(any(section.source_title == "Heading 1" for section in result.canonical_structure.sections))

    def test_g34_unresolved_macro_regions_do_not_create_full_document_calls(self):
        document = doc([block("page", i, 1, f"Page {i}") for i in range(1, 120)])
        recon = reconnaissance_from_json(recon_json([
            region(f"Semantic {i}", source=None, start=None) for i in range(1, 20)
        ]))
        candidates = discover_structural_candidates(document, recon, ground_macro_regions(document, recon))
        scopes = prepare_candidate_analysis_scopes(
            document,
            recon,
            ground_macro_regions(document, recon),
            candidates,
            max_model_calls=MAX_HIERARCHICAL_MODEL_CALLS - 1,
        )
        self.assertFalse(any(scope.block_count == len(document.blocks) for scope in scopes))
        self.assertTrue(all(scope.skipped for scope in scopes))

    def test_g34_candidate_failure_produces_partial_safely(self):
        document = doc([block("page", 1, 1, "Main", role=ROLE_HEADING)])
        client = FakeClient([
            recon_json([region("Main", source="Main", start=None)]),
            "{",
        ])
        result = interpret_hierarchical_structure(document=document, model="mock", client=client)
        self.assertTrue(any(local.interpretation_failed for local in result.local_structures))

    def test_g35a_missed_same_level_native_heading_becomes_macro_boundary(self):
        document = doc([
            block("document_flow", 1, 1, "Chapter A", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "A body"),
            block("document_flow", 1, 3, "Chapter B", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 4, "B body"),
            block("document_flow", 1, 5, "Chapter C", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([
            region("Chapter A", source="Chapter A", start=None),
            region("Chapter B", source="Chapter B", start=None),
        ]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertTrue(any(region.source_title == "Chapter C" for region in reconciled.reconnaissance.macro_regions))
        self.assertEqual(len(reconciled.diagnostics["accepted_candidate_macro_boundaries"]), 1)

    def test_g35a_nested_heading_inside_macro_does_not_split(self):
        document = doc([
            block("document_flow", 1, 1, "Chapter B", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Section B.1", role=ROLE_HEADING, style="Heading 2", hierarchy=2),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Chapter B", source="Chapter B", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertFalse(any(region.source_title == "Section B.1" for region in reconciled.reconnaissance.macro_regions))
        reasons = {item["rejection_reason"] for item in reconciled.diagnostics["rejected_candidate_macro_boundaries"]}
        self.assertIn("nested_below_current_macro_level", reasons)

    def test_g35a_short_heading_like_text_alone_does_not_split_macro(self):
        document = doc([
            block("page", 1, 1, "Main", role=ROLE_HEADING, hierarchy=1),
            block("page", 1, 2, "Short Phrase"),
        ])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertFalse(any(region.source_title == "Short Phrase" for region in reconciled.reconnaissance.macro_regions))

    def test_g35a_arbitrary_language_native_heading_splits_without_keyword_rules(self):
        document = doc([
            block("document_flow", 1, 1, "Parte Alfa", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Corpo"),
            block("document_flow", 1, 3, "Sezione Inventata", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Parte Alfa", source="Parte Alfa", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertTrue(any(region.source_title == "Sezione Inventata" for region in reconciled.reconnaissance.macro_regions))

    def test_g35a_duplicate_candidate_at_grounded_boundary_is_suppressed(self):
        document = doc([
            block("document_flow", 1, 1, "Chapter A", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Chapter B", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([
            region("Chapter A", source="Chapter A", start=None),
            region("Chapter B", source="Chapter B", start=None),
        ]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertEqual(reconciled.diagnostics["accepted_candidate_macro_boundaries"], [])
        self.assertGreaterEqual(reconciled.diagnostics["duplicate_proposals_suppressed"], 2)

    def test_g35a_semantic_unresolved_region_does_not_hide_physical_heading(self):
        document = doc([
            block("document_flow", 1, 1, "Chapter A", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Chapter B", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([
            region("Chapter A", source="Chapter A", start=None),
            region("Invented Semantic Region", source=None, start=None),
        ]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertTrue(any(region.source_title == "Chapter B" for region in reconciled.reconnaissance.macro_regions))
        self.assertGreaterEqual(reconciled.diagnostics["unresolved_semantic_macro_regions_preserved"], 1)

    def test_g35a_pdf_weak_metadata_does_not_aggressively_fragment(self):
        document = doc([
            block("page", 1, 1, "Main", role=ROLE_BODY),
            block("page", 1, 2, "Possible Heading", role=ROLE_BODY),
        ])
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        self.assertFalse(any(region.source_title == "Possible Heading" for region in reconciled.reconnaissance.macro_regions))

    def test_g35a_pptx_title_evidence_can_participate_but_not_every_slide_title(self):
        document = doc([
            block("slide", 1, 1, "Opening", role=ROLE_TITLE, hierarchy=1),
            block("slide", 2, 1, "Detail", role=ROLE_TITLE, hierarchy=2),
            block("slide", 3, 1, "New Unit", role=ROLE_TITLE, hierarchy=1),
        ], file_type="pptx", unit_type="slide")
        recon = reconnaissance_from_json(recon_json([region("Opening", source="Opening", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        reconciled = reconcile_macro_boundaries(document, recon, groundings, candidates)
        titles = [region.source_title for region in reconciled.reconnaissance.macro_regions]
        self.assertIn("New Unit", titles)
        self.assertNotIn("Detail", titles)

    def test_g35b_marker_plus_title_is_one_candidate_group(self):
        document = doc([
            block("document_flow", 1, 1, "Chapter A", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "4", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 3, "Thermodynamics", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Chapter A", source="Chapter A", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        groups = propose_candidate_groups(document, candidates)
        compound = next(group for group in groups if group.relationship_hypothesis == "compound_heading_components")
        self.assertEqual([c.source_title for c in candidates if c.candidate_id in compound.candidate_ids], ["4", "Thermodynamics"])
        self.assertEqual(next(c.source_title for c in candidates if c.candidate_id == compound.primary_candidate_id), "Thermodynamics")

    def test_g35b_group_id_reconciles_to_one_logical_node_with_primary_anchor(self):
        document = doc([
            block("document_flow", 1, 1, "Chapter A", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "4", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 3, "Thermodynamics", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Chapter A", source="Chapter A", start=None)]))
        groundings = ground_macro_regions(document, recon)
        candidates = discover_structural_candidates(document, recon, groundings)
        groups = propose_candidate_groups(document, candidates)
        scopes = prepare_candidate_analysis_scopes(
            document,
            recon,
            groundings,
            candidates,
            groups,
            max_model_calls=MAX_HIERARCHICAL_MODEL_CALLS,
        )
        compound = next(group for group in groups if group.relationship_hypothesis == "compound_heading_components")
        scope = next(scope for scope in scopes if any(group.group_id == compound.group_id for group in scope.candidate_groups))
        local = validate_local_candidate_sections(
            scope,
            local_structure_from_json(
                scope.region.region_id,
                compact_local_json([
                    {**compact_boundary("4 Thermodynamics"), "group_id": compound.group_id},
                ]),
            ),
        )
        canonical = reconcile_hierarchy(document, recon, [local])
        sections = [section for section in canonical.sections if section.source_title == "Thermodynamics"]
        self.assertEqual(len(sections), 1)
        self.assertEqual(
            sections[0].expected_start,
            next(c.source_position for c in candidates if c.candidate_id == compound.primary_candidate_id),
        )
        self.assertEqual(canonical.diagnostics["multi_component_logical_nodes"], 1)

    def test_g35b_sibling_headings_are_not_blindly_merged(self):
        document = doc([
            block("document_flow", 1, 1, "Main", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Methods", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 3, "Results", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Main", source="Main", start=None)]))
        groups = propose_candidate_groups(document, discover_structural_candidates(document, recon, ground_macro_regions(document, recon)))
        self.assertFalse(any(len(group.candidate_ids) > 1 for group in groups))

    def test_g35b_parent_child_hierarchy_is_not_compound_merged(self):
        document = doc([
            block("document_flow", 1, 1, "Introduction", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "Background", role=ROLE_HEADING, style="Heading 2", hierarchy=2),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("Introduction", source="Introduction", start=None)]))
        groups = propose_candidate_groups(document, discover_structural_candidates(document, recon, ground_macro_regions(document, recon)))
        self.assertFalse(any(len(group.candidate_ids) > 1 for group in groups))

    def test_g35b_three_component_stack_is_not_collapsed_into_one_pairwise_merge(self):
        document = doc([
            block("document_flow", 1, 1, "PART II", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 2, "CHAPTER 4", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
            block("document_flow", 1, 3, "Thermodynamics", role=ROLE_HEADING, style="Heading 1", hierarchy=1),
        ], file_type="docx", unit_type="document_flow")
        recon = reconnaissance_from_json(recon_json([region("PART II", source="PART II", start=None)]))
        groups = propose_candidate_groups(document, discover_structural_candidates(document, recon, ground_macro_regions(document, recon)))
        multi = [group for group in groups if len(group.candidate_ids) > 1]
        self.assertLessEqual(max((len(group.candidate_ids) for group in groups), default=1), 2)
        self.assertFalse(any("cand-00001" in group.candidate_ids and len(group.candidate_ids) > 1 for group in groups))
        self.assertTrue(all(group.relationship_hypothesis in {"single_structural_component", "compound_heading_components"} for group in groups))
        self.assertLessEqual(len(multi), 1)

    def test_g35b_macro_local_same_anchor_same_title_is_suppressed(self):
        document = doc([block("page", 1, 1, "CAPITOLO II\nBody")])
        recon = reconnaissance_from_json(recon_json([region("CAPITOLO II", source="CAPITOLO II")]))
        local = local_structure_from_json("region-0001", local_json([local_section("CAPITOLO II")]))
        canonical = reconcile_hierarchy(document, recon, [local])
        self.assertEqual(len([section for section in canonical.sections if section.source_title == "CAPITOLO II"]), 1)
        self.assertEqual(canonical.diagnostics["macro_local_duplicate_nodes_suppressed"], 1)

    def test_g35b_pptx_title_subtitle_is_ambiguous_not_forced_hierarchy(self):
        document = doc([
            block("slide", 1, 1, "Energy Systems", role=ROLE_TITLE, hierarchy=1),
            block("slide", 1, 2, "A conceptual overview", role=ROLE_HEADING, hierarchy=1),
        ], file_type="pptx", unit_type="slide")
        recon = reconnaissance_from_json(recon_json([region("Energy Systems", source="Energy Systems", start=None)]))
        groups = propose_candidate_groups(document, discover_structural_candidates(document, recon, ground_macro_regions(document, recon)))
        self.assertTrue(any(group.relationship_hypothesis == "ambiguous_grouping" for group in groups))

    def test_g35b_pdf_weak_adjacency_does_not_force_compound_group(self):
        document = doc([
            block("page", 1, 1, "4", role=ROLE_BODY),
            block("page", 1, 2, "Thermodynamics", role=ROLE_BODY),
        ])
        recon = reconnaissance_from_json(recon_json([region("Thermodynamics", source="Thermodynamics", start=None)]))
        groups = propose_candidate_groups(document, discover_structural_candidates(document, recon, ground_macro_regions(document, recon)))
        self.assertFalse(any(len(group.candidate_ids) > 1 for group in groups))


if __name__ == "__main__":
    unittest.main()
