import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from taxonomy_assignment import validate_chapter_coverage


class ChapterCoverageDiagnosticsTests(unittest.TestCase):
    def check_report(self, candidates, chunks, ownership, fails):
        before = copy.deepcopy((candidates, chunks, ownership))
        output = io.StringIO()
        with redirect_stdout(output):
            if fails:
                with self.assertRaisesRegex(ValueError, 'validated source coverage'):
                    validate_chapter_coverage(candidates, chunks, ownership)
            else:
                validate_chapter_coverage(candidates, chunks, ownership)
        self.assertEqual(before, (candidates, chunks, ownership))
        return json.loads(output.getvalue().split(' ', 1)[1])

    def test_unsupported_candidate_does_not_acquire_batch_evidence(self):
        report = self.check_report([
            {'topic': 'Unsupported', 'chapter_id': 'a', 'discovery_chunk_ids': [1], 'evidence_ids': []},
            {'topic': 'Supported', 'chapter_id': 'a', 'evidence_ids': [1]}],
            [{'chunk_id': 1, 'page': 3}], {1: 'a'}, True)
        self.assertEqual(len(report['unsupported_candidates']), 1)
        self.assertEqual(report['uncovered_chunks'], [])
        self.assertEqual(report['unsupported_candidates'][0]['discovery_chunk_ids'], [1])

    def test_uncovered_chunk_reports_page_and_scope(self):
        report = self.check_report([{'chapter_id': 'a', 'evidence_ids': [1]}],
            [{'chunk_id': 1, 'page': 3}, {'chunk_id': 2, 'page': 4}], {1: 'a', 2: 'review'}, True)
        self.assertEqual(report['unsupported_candidates'], [])
        self.assertEqual(report['uncovered_chunks'], [{'chunk_id': 2, 'page': 4, 'chapter_id': 'review'}])

    def test_cross_chapter_and_unknown_links_rejected(self):
        report = self.check_report([{'chapter_id': 'a', 'evidence_ids': [1, 2, 99]}],
            [{'chunk_id': 1}, {'chunk_id': 2}], {1: 'a', 2: 'b'}, True)
        self.assertEqual(len(report['invalid_links']), 2)

    def test_complete_coverage_preserves_data(self):
        report = self.check_report([{'chapter_id': 'a', 'evidence_ids': [1]},
            {'chapter_id': 'review', 'evidence_ids': [2]}],
            [{'chunk_id': 1}, {'chunk_id': 2}], {1: 'a', 2: 'review'}, False)
        self.assertEqual(report['covered_chunks'], 2)

class SourceRescueTests(unittest.TestCase):
    def plan(self, scores, discovery=(1, 2), owners=None):
        from types import SimpleNamespace
        from unittest.mock import patch
        from taxonomy_assignment import plan_assignments
        candidate = dict(assignment_id='topic', topic='Subject', source_section=None,
                         embedding=[1.0], chapter_id='a', discovery_chunk_ids=list(discovery))
        rows = [(i, 'source', '', 'topic', 'Subject', None, score) for i, score in scores.items()]
        db = SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(fetchall=lambda: ([(r[0], r[1], r[2]) for r in rows] if "cross join" not in str(args[0]) else [(r[0], r[3], r[6]) for r in rows])))
        with patch('taxonomy_assignment.source_snapshot', return_value={i: str(i) for i in scores}):
            links, _ = plan_assignments(db, [candidate], 'project', lambda *args: args[-1],
                                       chapter_ownership=owners)
        return links['topic']

    def test_existing_topic_link_does_not_prevent_source_rescue(self):
        self.assertEqual(self.plan({1: .8, 2: .5}, owners={1: 'a', 2: 'a'}), {1, 2})

    def test_weak_source_remains_uncovered(self):
        self.assertEqual(self.plan({1: .8, 2: .45}, owners={1: 'a', 2: 'a'}), {1})

    def test_batch_membership_alone_is_not_evidence(self):
        self.assertEqual(self.plan({1: .8, 2: None}, owners={1: 'a', 2: 'a'}), {1})

    def test_rescue_does_not_cross_chapter_or_batch(self):
        self.assertEqual(self.plan({1: .8, 2: .5}, owners={1: 'a', 2: 'b'}), {1})
        self.assertEqual(self.plan({1: .8, 2: .5}, discovery=(1,), owners={1: 'a', 2: 'a'}), {1})

    def test_legacy_assignment_is_unchanged(self):
        self.assertEqual(self.plan({1: .8, 2: .5}), {1})

class DiscoveryCheckpointTests(unittest.TestCase):
    def test_opt_in_atomic_private_snapshot(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from taxonomy_assignment import save_discovery_checkpoint
        candidates = [{'topic': 'Original', 'description': 'Evidence', 'discovery_chunk_ids': [1]}]
        before = copy.deepcopy(candidates)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'DOUNO_DISCOVERY_CHECKPOINT_DIR': directory}):
                with redirect_stdout(io.StringIO()):
                    path = Path(save_discovery_checkpoint(candidates, {1: 'hash'}, {'mode': 'index'},
                                                          'p', 'm', 'd', {1: 'chapter'}))
            self.assertEqual(json.loads(path.read_text())['candidates'], candidates)
            self.assertEqual(json.loads(path.read_text())['source_snapshot'], {'1': 'hash'})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
        self.assertEqual(candidates, before)

    def test_disabled_without_configuration(self):
        from unittest.mock import patch
        from taxonomy_assignment import save_discovery_checkpoint
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(save_discovery_checkpoint([], {}, {}, 'p', 'm', 'd', None))
