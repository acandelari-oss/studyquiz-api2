"""Offline tests: extract pure production functions with AST, never import main."""
import ast
import contextlib
import io
from pathlib import Path
import random
import re
import sys
import unittest
from dataclasses import asdict, replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canonical_document import CanonicalBlock, SourcePosition
from document_hierarchy_contracts import CanonicalDocumentNode, CanonicalDocumentStructure, SourceSpan
from source_chunk_alignment import SourceBinding, align_chunks, section_extents


def production_functions():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ('clean_text', 'chunk_text')]
    assert len(funcs) == 2
    namespace = {'re': re}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<production-functions>', 'exec'), namespace)
    return namespace['clean_text'], namespace['chunk_text']


CLEAN, CHUNK = production_functions()


def block(text, index=1):
    return CanonicalBlock(f'b{index}', text, text, index, SourcePosition('page', 1, index, 0, len(text)))


def run(text, bindings, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return align_chunks('doc', text, bindings, clean=CLEAN, chunk=CHUNK, **kwargs)


def node(nid, b, title_length, level=1, parent=None):
    return CanonicalDocumentNode(nid, b.raw_text[:title_length], None, level, parent, b.source_order,
        SourceSpan(replace(b.source_position, end_offset=None), replace(b.source_position, start_offset=title_length, end_offset=None)),
        'explicit_source', 'HIGH', 'accepted')


class AlignmentTests(unittest.TestCase):
    def test_preserves_actual_pipeline_output_randomized_and_overlaps(self):
        rng = random.Random(41)
        fixtures = ['', 'A'*1000, 'A'*1001, '  chapterOne 12B.Test\n\nnext\tword ', 'é\u0301漢字\r\nX']
        fixtures += [''.join(rng.choices('abAZ09 .\t\n-é', k=n)) for n in range(0, 3000, 47)]
        for text in fixtures:
            with self.subTest(length=len(text)):
                result = run(text, [SourceBinding('doc', 0, len(text), block(text))] if text else [])
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = [CLEAN(c) for c in CHUNK(CLEAN(text)) if CLEAN(c)]
                self.assertEqual([c.text for c in result.chunks], expected)
                self.assertNotIn('transformation_mismatch', result.diagnostics)
                self.assertFalse(set(result.diagnostics) & {'unresolved', 'ambiguous'})
                for c in result.chunks:
                    for interval in c.intervals:
                        if interval.status == 'exact':
                            raw = ''.join(text[s.start.start_offset:s.end.start_offset] for s in interval.source_spans)
                            self.assertEqual(' ' if raw.isspace() else raw, c.text[interval.chunk_start:interval.chunk_end])

    def test_cross_block_and_generated_separator(self):
        a, b = block('chapterOne'), block('Body2', 2)
        text = a.raw_text + b.raw_text
        r = run(text, [SourceBinding('doc', 0, 10, a), SourceBinding('doc', 10, len(text), b)])
        self.assertEqual(r.chunks[0].text, 'chapter One Body 2')
        self.assertEqual({s.start.block_index for i in r.chunks[0].intervals for s in i.source_spans}, {1, 2})
        self.assertTrue(any(i.status == 'generated' and not i.source_spans for i in r.chunks[0].intervals))
        self.assertEqual(r.diagnostics, ())
        self.assertEqual(asdict(r)['contract_version'], 1)

    def test_whitespace_can_map_multiple_blocks(self):
        a, b = block('A '), block('\nB', 2)
        r = run('A \nB', [SourceBinding('doc', 0, 2, a), SourceBinding('doc', 2, 4, b)])
        gap = r.chunks[0].intervals[1]
        self.assertEqual(gap.status, 'exact')
        self.assertEqual(len(gap.source_spans), 2)

    def test_repeated_text_is_not_matched_by_guessing(self):
        text = 'repeat repeat repeat'
        r = run(text, [SourceBinding('doc', 0, len(text), block(text))], max_chars=8, overlap=2)
        self.assertEqual(r.chunks[1].intervals[0].source_spans[0].start.start_offset, 7)

    def test_ambiguous_missing_and_ocr_offsets(self):
        text = 'same text'
        a, b = block(text), block(text, 2)
        r = run(text, [SourceBinding('doc', 0, len(text), a), SourceBinding('doc', 0, len(text), b)])
        self.assertTrue(all(i.status == 'ambiguous' and not i.source_spans for i in r.chunks[0].intervals))
        for bindings in ([], [SourceBinding('doc', 0, len(text), a, reliable=False)], [SourceBinding('doc', 0, len(text), block('sarne text'))]):
            r = run(text, bindings)
            self.assertTrue(all(i.status == 'unresolved' and not i.source_spans for i in r.chunks[0].intervals))

    def test_pdf_fragments_require_explicit_verified_bindings(self):
        text = 'First body Second body'
        a, b = block('First body'), block('Second body', 2)
        r = run(text, [SourceBinding('doc', 0, 10, a), SourceBinding('doc', 11, 22, b)])
        self.assertEqual(r.chunks[0].intervals[10].status, 'unresolved')
        self.assertEqual(r.chunks[0].intervals[11].source_spans[0].start.block_index, 2)
        self.assertEqual(r.chunks[0].intervals[11].source_spans[0].start.start_offset, 0)

    def test_isolation_and_contract_drift(self):
        with self.assertRaisesRegex(ValueError, 'Cross-document'):
            run('a', [SourceBinding('other', 0, 1, block('a'))])
        r = align_chunks('doc', 'a', [], clean=lambda x: x.upper(), chunk=lambda x, **kw: [x])
        self.assertEqual(r.chunks[0].text, 'A')
        self.assertEqual(r.diagnostics, ('transformation_mismatch',))
        self.assertEqual(r.chunks[0].intervals[0].source_spans, ())

    def test_structured_nested_heading_and_full_extents(self):
        blocks = [block('Chapter A body', 1), block('Minor detail body', 2), block('Chapter B body', 3)]
        nodes = [node('A', blocks[0], 9), node('minor', blocks[1], 5, 2, 'A'), node('B', blocks[2], 9)]
        structure = CanonicalDocumentStructure('doc', 'pdf', 'sample', 'accepted', 'HIGH', nodes)
        extents = section_extents(structure, blocks, ordering_validated=True)
        self.assertEqual(extents[0].heading_span.end.start_offset, 9)
        self.assertEqual(extents[0].extent.end.block_index, 3)
        self.assertEqual(extents[1].parent_id, 'A')
        self.assertEqual(extents[1].extent.end.block_index, 3)
        self.assertEqual(extents[2].extent.end.start_offset, len(blocks[2].raw_text))
        # Two headings on the same page remain distinct; titles/keywords are irrelevant.
        self.assertEqual(len(extents), 3)
        for bad in (replace(structure, structure_confidence='LOW'), replace(structure, nodes=[nodes[0], replace(nodes[1], parent_id='B'), nodes[2]])):
            with self.assertRaises(ValueError):
                section_extents(bad, blocks, ordering_validated=True)
        with self.assertRaises(ValueError):
            section_extents(structure, blocks)

    def test_nonzero_source_offsets_and_conflicting_coordinates(self):
        b = block('alpha')
        b = replace(b, source_position=replace(b.source_position, start_offset=12, end_offset=17))
        r = run('alpha', [SourceBinding('doc', 0, 5, b)])
        self.assertEqual(r.chunks[0].intervals[0].source_spans[0].start.start_offset, 12)
        other = replace(b, block_id='another')
        with self.assertRaisesRegex(ValueError, 'Ambiguous canonical'):
            run('alpha', [SourceBinding('doc', 0, 5, b), SourceBinding('doc', 0, 5, other)])

    def test_overlapping_heading_spans_rejected(self):
        b = block('First Second Body')
        a = node('a', b, 10)
        child = replace(node('child', b, 12, 2, 'a'), source_order=2,
                        source_span=SourceSpan(replace(b.source_position, start_offset=6),
                                               replace(b.source_position, start_offset=12)))
        s = CanonicalDocumentStructure('doc', 'pdf', 'sample', 'accepted', 'HIGH', [a, child])
        with self.assertRaisesRegex(ValueError, 'Overlapping heading'):
            section_extents(s, [b], ordering_validated=True)

    def test_unknown_heading_extent_is_not_fabricated(self):
        b = block('Heading body')
        n = node('a', b, 7)
        for changed in (replace(n, source_span=SourceSpan(n.source_span.start)), replace(n, parent_id='missing'), replace(n, status='unresolved')):
            s = CanonicalDocumentStructure('doc', 'pdf', 'sample', 'accepted', 'HIGH', [changed])
            with self.assertRaises(ValueError):
                section_extents(s, [b], ordering_validated=True)


if __name__ == '__main__':
    unittest.main()
