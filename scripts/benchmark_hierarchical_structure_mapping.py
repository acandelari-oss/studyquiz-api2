#!/usr/bin/env python3
"""Isolated hierarchical structure-mapping benchmark for Sprint G3."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import adapt_document
from hierarchical_structure_interpreter import (
    GLOBAL_RECONNAISSANCE_SCHEMA,
    LOCAL_STRUCTURE_SCHEMA,
    UniversalStructureError,
    interpret_hierarchical_structure,
    strict_schema_violations,
)


def load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hierarchical universal document structure mapping."
    )
    parser.add_argument("file", help="Path to PDF, DOCX, or PPTX.")
    parser.add_argument("--model", required=True, help="OpenAI model ID.")
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv_if_present(Path(args.env_file))
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is unavailable.", file=sys.stderr)
        return 2

    schema_errors = (
        strict_schema_violations(GLOBAL_RECONNAISSANCE_SCHEMA)
        + strict_schema_violations(LOCAL_STRUCTURE_SCHEMA)
    )
    if schema_errors:
        print("ERROR: strict schema validation failed before API call.", file=sys.stderr)
        for error in schema_errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2

    file_bytes = path.read_bytes()
    document = adapt_document(file_bytes, path.name)
    print_header(document)

    try:
        result = interpret_hierarchical_structure(
            document=document,
            original_file_bytes=file_bytes if document.file_type == "pdf" else None,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    except UniversalStructureError as exc:
        print("GLOBAL / LOCAL MODEL INTERPRETATION")
        print(f"failed: {exc}")
        return 1

    print_global_reconnaissance(result)
    print_macro_region_grounding(result)
    print_candidate_discovery(result)
    print_candidate_grouping(result)
    print_macro_boundary_reconciliation(result)
    print_local_analysis(result)
    print_canonical_structure(result)
    print_logical_reconciliation(result)
    print_quality(result)
    print_mapping(result, verbose=args.verbose)
    print_usage(result)
    return 0


def print_header(document) -> None:
    print("================================================")
    print("HIERARCHICAL STRUCTURE MAPPING BENCHMARK")
    print("================================================")
    print("DOCUMENT")
    print(f"file: {document.filename}")
    print(f"type: {document.file_type}")
    print(f"units: {len(document.units)}")
    print(f"blocks: {len(document.blocks)}")
    print()


def print_global_reconnaissance(result) -> None:
    recon = result.reconnaissance
    print("GLOBAL RECONNAISSANCE")
    print(f"document_title: {recon.document_title or 'n/a'}")
    print(f"document_type: {recon.document_type or 'n/a'}")
    print(f"structure_confidence: {recon.structure_confidence}")
    print(f"organization_style: {recon.organization_style}")
    print(f"macro_regions: {len(recon.macro_regions)}")
    for region in recon.macro_regions:
        start = _position(region.approximate_start)
        end = _position(region.approximate_end)
        print(
            f"- {region.region_id} | {region.semantic_title} | "
            f"confidence={region.confidence} | start={start} | end={end}"
        )
    print()


def print_local_analysis(result) -> None:
    print("LOCAL ANALYSIS")
    print(f"semantic macro-regions: {len(result.reconnaissance.macro_regions)}")
    print(f"local scopes analyzed/skipped: {len(result.local_structures)}")
    per_region = {}
    for local in result.local_structures:
        diagnostics = local.diagnostics or {}
        macro_id = diagnostics.get("macro_region_id") or local.region_id
        bucket = per_region.setdefault(
            macro_id,
            {"scopes": 0, "failed": 0, "tokens_in": 0, "tokens_out": 0, "sections": 0},
        )
        bucket["scopes"] += 1
        bucket["failed"] += 1 if local.interpretation_failed else 0
        bucket["tokens_in"] += diagnostics.get("input_tokens") or 0
        bucket["tokens_out"] += diagnostics.get("output_tokens") or 0
        bucket["sections"] += len(local.sections)
    for local in result.local_structures:
        diagnostics = local.diagnostics or {}
        print(
            f"- {diagnostics.get('macro_region_id', local.region_id)} / "
            f"scope-{diagnostics.get('subscope_index', 1)} | "
            f"type={diagnostics.get('scope_type', 'n/a')} | "
            f"reason={diagnostics.get('subdivision_reason', 'n/a')} | "
            f"blocks={diagnostics.get('block_count', 'n/a')} | "
            f"chars={diagnostics.get('input_char_count', 'n/a')} | "
            f"candidates={diagnostics.get('candidate_count', 'n/a')} | "
            f"range={diagnostics.get('source_start')} -> {diagnostics.get('source_end')} | "
            f"boundaries={diagnostics.get('structural_boundaries', len(local.sections))} | "
            f"continuations={len(local.continuation_nodes)} | "
            f"local_only={len(local.local_only_nodes)} | "
            f"failed={local.interpretation_failed}"
        )
        if diagnostics.get("skipped"):
            print(f"  skipped: {diagnostics.get('skip_reason')}")
        if diagnostics.get("failure_reason"):
            print(f"  failure: {diagnostics.get('failure_reason')}")
        if diagnostics.get("response_status") or diagnostics.get("incomplete_reason"):
            print(
                "  response: "
                f"status={diagnostics.get('response_status')} | "
                f"incomplete={diagnostics.get('incomplete_reason')} | "
                f"raw_chars={diagnostics.get('raw_output_chars')}"
            )
    print("PER MACRO-REGION SUMMARY")
    for macro_id, summary in sorted(per_region.items()):
        print(
            f"- {macro_id} | scopes={summary['scopes']} | failed={summary['failed']} | "
            f"sections={summary['sections']} | input={summary['tokens_in']} | "
            f"output={summary['tokens_out']}"
        )
    print()


def print_macro_region_grounding(result) -> None:
    print("MACRO REGION GROUNDING")
    for grounding in result.groundings:
        if grounding.source_position:
            location = (
                f"{grounding.source_position.unit_type}:"
                f"{grounding.source_position.unit_index}:"
                f"{grounding.source_position.block_index}@"
                f"{grounding.source_position.start_offset}"
            )
        else:
            location = "unresolved"
        print(
            f"- {grounding.region_id} | {grounding.status} | "
            f"method={grounding.method} | fidelity={grounding.fidelity} | "
            f"confidence={grounding.confidence} | {location}"
        )
        if grounding.reason:
            print(f"  reason: {grounding.reason}")
    print()


def print_candidate_discovery(result) -> None:
    diagnostics = result.diagnostics.get("candidate_diagnostics") or {}
    print("CANDIDATE DISCOVERY")
    print(f"total candidates: {diagnostics.get('total_candidates', 'n/a')}")
    print(f"candidate density/block: {diagnostics.get('candidate_density_per_block', 'n/a')}")
    print(f"retained for model analysis: {diagnostics.get('candidates_retained_for_model_analysis', 'n/a')}")
    print(f"candidate analysis calls: {diagnostics.get('candidate_analysis_calls', 'n/a')}")
    print(f"max candidates/call: {diagnostics.get('max_candidates_in_one_call', 'n/a')}")
    print(f"call budget exhausted: {diagnostics.get('call_budget_exhausted', False)}")
    print(f"skipped candidate scopes: {diagnostics.get('skipped_candidate_scopes', 0)}")
    print("candidates by evidence:")
    for signal, count in sorted((diagnostics.get("candidates_by_evidence_type") or {}).items()):
        print(f"- {signal}: {count}")
    print()


def print_candidate_grouping(result) -> None:
    diagnostics = result.diagnostics.get("candidate_grouping_diagnostics") or {}
    print("CANDIDATE GROUPING")
    print(f"group proposals: {diagnostics.get('group_proposals', 'n/a')}")
    print(f"single-candidate proposals: {diagnostics.get('single_candidate_proposals', 'n/a')}")
    print(f"multi-candidate proposals: {diagnostics.get('multi_candidate_proposals', 'n/a')}")
    print(f"ambiguous proposals: {diagnostics.get('ambiguous_proposals', 'n/a')}")
    print(f"rejected proposals: {diagnostics.get('rejected_proposals', 'n/a')}")
    for group in diagnostics.get("multi_component_groups") or []:
        print(
            f"- {group.get('group_id')} | {group.get('relationship_hypothesis')} | "
            f"primary={group.get('primary_candidate_id')} | "
            f"candidates={','.join(group.get('candidate_ids') or [])} | "
            f"confidence={group.get('confidence')}"
        )
    print()


def print_macro_boundary_reconciliation(result) -> None:
    diagnostics = result.diagnostics.get("macro_boundary_reconciliation") or {}
    print("MACRO BOUNDARY RECONCILIATION")
    print(f"macro regions before: {len(diagnostics.get('macro_regions_before') or [])}")
    print(f"macro regions after: {len(diagnostics.get('macro_regions_after') or [])}")
    print(f"accepted candidate macro boundaries: {len(diagnostics.get('accepted_candidate_macro_boundaries') or [])}")
    print(f"rejected candidate macro boundaries: {len(diagnostics.get('rejected_candidate_macro_boundaries') or [])}")
    print(f"physical scope changes: {diagnostics.get('physical_scope_changes', 0)}")
    print(f"duplicate proposals suppressed: {diagnostics.get('duplicate_proposals_suppressed', 0)}")
    print(f"unresolved semantic macro-regions preserved: {diagnostics.get('unresolved_semantic_macro_regions_preserved', 0)}")
    for proposal in diagnostics.get("accepted_candidate_macro_boundaries") or []:
        print(
            f"- accepted {proposal.get('candidate_id')} | {proposal.get('source_title')} | "
            f"position={proposal.get('source_position')} | split={proposal.get('macro_region_id')} | "
            f"evidence={','.join(proposal.get('evidence_signals') or [])}"
        )
    for proposal in (diagnostics.get("rejected_candidate_macro_boundaries") or [])[:10]:
        print(
            f"- rejected {proposal.get('candidate_id')} | {proposal.get('source_title')} | "
            f"reason={proposal.get('rejection_reason')}"
        )
    print()


def print_canonical_structure(result) -> None:
    print("CANONICAL STRUCTURE")
    lookup = {section.section_id: section for section in result.canonical_structure.sections}
    for section in result.canonical_structure.sections:
        indent = "  " * max(0, section.level - 1)
        parent = lookup.get(section.parent_id or "")
        parent_label = f" parent={parent.semantic_title}" if parent else ""
        print(
            f"{indent}- {section.source_title or section.semantic_title} "
            f"[{section.structural_role} | {section.confidence}{parent_label}]"
        )
    print()


def print_logical_reconciliation(result) -> None:
    diagnostics = result.canonical_structure.diagnostics or {}
    print("LOGICAL STRUCTURE RECONCILIATION")
    print(f"logical nodes: {diagnostics.get('logical_nodes', 0)}")
    print(f"multi-component logical nodes: {diagnostics.get('multi_component_logical_nodes', 0)}")
    print(f"primary G2 anchor nodes: {diagnostics.get('primary_g2_anchor_nodes', 0)}")
    print(f"macro/local duplicates suppressed: {diagnostics.get('macro_local_duplicate_nodes_suppressed', 0)}")
    print(f"duplicate canonical paths: {diagnostics.get('duplicate_canonical_paths', 0)}")
    print()


def print_quality(result) -> None:
    quality = result.canonical_structure.structure_quality
    fidelity = result.mapping_fidelity
    print("STRUCTURE QUALITY")
    print(f"status: {quality.status}")
    print(f"score: {quality.score}")
    for reason in quality.reasons:
        print(f"- {reason}")
    print()
    print("MAPPING FIDELITY")
    print(f"status: {fidelity.status}")
    print(f"score: {fidelity.score}")
    for reason in fidelity.reasons:
        print(f"- {reason}")
    print()


def print_mapping(result, *, verbose: bool = False) -> None:
    mapping = result.mapping_result
    print("ANCHOR RESULTS")
    print(f"anchored: {mapping.sections_anchored}")
    print(f"unresolved: {mapping.unresolved_sections}")
    print(f"exact: {mapping.exact_count}")
    print(f"normalized: {mapping.normalized_count}")
    print(f"fuzzy: {mapping.fuzzy_count}")
    print(f"model_only: {mapping.model_only_count}")
    print()
    print("SEGMENTS")
    for index, segment in enumerate(mapping.segments, start=1):
        preview = " ".join(segment.text.split())[:140]
        print(f"{index:03d} | section={segment.section_path} | {preview}")
        if verbose:
            print(segment.raw_text)
    print()
    print("CHUNKS")
    for index, chunk in enumerate(mapping.diagnostic_chunks, start=1):
        preview = " ".join(chunk.text.split())[:140]
        print(f"{index:03d} | section={chunk.section_path} | {preview}")
    print()
    print("VALIDATION")
    print(f"boundary crossing: {mapping.diagnostics.get('chunks_crossing_boundaries', 0)}")
    print(f"collisions: {mapping.duplicate_anchor_collisions}")
    print(f"order violations: {mapping.source_order_violations}")
    print(f"duplicate canonical paths: {result.canonical_structure.diagnostics.get('duplicate_canonical_paths', 0)}")
    print(f"overlapping regions: {result.canonical_structure.diagnostics.get('overlapping_region_boundaries', 0)}")
    print(f"unresolved: {mapping.unresolved_sections}")
    print()


def print_usage(result) -> None:
    print("MODEL USAGE")
    local_calls = sum(1 for local in result.local_structures if not (local.diagnostics or {}).get("skipped"))
    candidate_calls = sum(1 for local in result.local_structures if (local.diagnostics or {}).get("scope_type") == "candidate_batch" and not (local.diagnostics or {}).get("skipped"))
    fallback_calls = sum(1 for local in result.local_structures if (local.diagnostics or {}).get("scope_type") == "fallback_window")
    skipped = sum(1 for local in result.local_structures if (local.diagnostics or {}).get("skipped"))
    grounded_regions = sum(1 for grounding in result.groundings if grounding.status == "grounded")
    unresolved_regions = sum(1 for grounding in result.groundings if grounding.status != "grounded")
    subdivided_regions = len({
        (local.diagnostics or {}).get("macro_region_id")
        for local in result.local_structures
        if (local.diagnostics or {}).get("subdivision_reason") == "local_scope_budget_exceeded"
    })
    failed_scopes = sum(1 for local in result.local_structures if local.interpretation_failed and not (local.diagnostics or {}).get("skipped"))
    successful_local_scopes = sum(
        1
        for local in result.local_structures
        if not local.interpretation_failed and not (local.diagnostics or {}).get("skipped")
    )
    total_boundaries = sum(len(local.sections) for local in result.local_structures)
    successful_output_tokens = [
        (local.diagnostics or {}).get("output_tokens")
        for local in result.local_structures
        if not local.interpretation_failed
        and not (local.diagnostics or {}).get("skipped")
        and (local.diagnostics or {}).get("output_tokens") is not None
    ]
    successful_boundaries = [
        len(local.sections)
        for local in result.local_structures
        if not local.interpretation_failed and not (local.diagnostics or {}).get("skipped")
    ]
    print("global model calls: 1")
    print(f"semantic macro-regions: {len(result.reconnaissance.macro_regions)}")
    print(f"grounded macro-regions: {grounded_regions}")
    print(f"unresolved macro-regions: {unresolved_regions}")
    print(f"local model calls: {local_calls}")
    print(f"candidate-analysis calls: {candidate_calls}")
    print(f"subdivided macro-regions: {subdivided_regions}")
    print(f"successful local scopes: {successful_local_scopes}")
    print(f"failed local scopes: {failed_scopes}")
    print(f"total local structural boundaries emitted: {total_boundaries}")
    print(f"fallback-window calls: {fallback_calls}")
    print(f"skipped unresolved regions: {skipped}")
    print(f"total calls: {1 + local_calls}")
    total_input = 0
    total_output = 0
    total_tokens = 0
    all_diagnostics = [result.diagnostics.get("global_diagnostics")]
    all_diagnostics.extend(
        local.diagnostics for local in result.local_structures if local.diagnostics
    )
    for diagnostics in all_diagnostics:
        if not diagnostics:
            continue
        if isinstance(diagnostics, dict):
            total_input += diagnostics.get("input_tokens") or 0
            total_output += diagnostics.get("output_tokens") or 0
            total_tokens += diagnostics.get("total_tokens") or 0
        else:
            total_input += diagnostics.input_tokens or 0
            total_output += diagnostics.output_tokens or 0
            total_tokens += diagnostics.total_tokens or 0
    print(f"input_tokens: {total_input or 'n/a'}")
    print(f"output_tokens: {total_output or 'n/a'}")
    print(f"total_tokens: {total_tokens or 'n/a'}")
    if successful_output_tokens:
        avg_output = sum(successful_output_tokens) / len(successful_output_tokens)
        print(f"avg local output tokens/successful scope: {avg_output:.1f}")
        print(f"max local output tokens: {max(successful_output_tokens)}")
    else:
        print("avg local output tokens/successful scope: n/a")
        print("max local output tokens: n/a")
    if successful_boundaries:
        avg_boundaries = sum(successful_boundaries) / len(successful_boundaries)
        print(f"avg structural boundaries/successful scope: {avg_boundaries:.1f}")
    else:
        print("avg structural boundaries/successful scope: n/a")
    print(f"elapsed_seconds: {result.elapsed_seconds:.2f}")
    print("================================================")


def _position(position) -> str:
    if not position:
        return "n/a"
    return (
        f"{position.unit_type}:{position.unit_index}:"
        f"{position.block_index}@{position.start_offset}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
