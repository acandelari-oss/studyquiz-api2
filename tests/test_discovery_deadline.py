"""Offline discovery budget tests: no application import, database or provider calls."""
import ast
import io
import json
import os
import time
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import taxonomy_coverage as coverage

TREE = ast.parse(Path('main.py').read_text())
TASK = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == 'process_topics_task')
VALID = {'categories': [{'name': 'Area', 'topics': [{'title': 'Topic', 'description': 'Evidence'}]}]}

def reply():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(VALID)))])

class DiscoveryDeadlineTests(unittest.TestCase):
    def test_complete_prompt_budget_splits_without_dropping_chunks(self):
        chunks = [{'chunk_id': i, 'text': 'x' * 1000} for i in range(40)]
        def prompt(group):
            return 'instructions/index' * 2000 + '\n\n'.join(coverage.discovery_source_text(c) for c in group)
        batches = list(coverage.bounded_batches(chunks, prompt_builder=prompt))
        self.assertGreater(len(batches), 2)
        self.assertEqual([c for b in batches for c in b], chunks)
        self.assertTrue(all(len(prompt(b)) <= coverage.REQUEST_MAX_CHARS for b in batches))

    def test_real_discovery_prompt_is_used_for_planning_and_request(self):
        builder = next(n for n in ast.walk(TASK) if isinstance(n, ast.FunctionDef)
                       and n.name == 'build_discovery_prompt')
        ns = dict(chapter_ownership=None, discovery_source_text=coverage.discovery_source_text,
                  language_instruction='English', module_organization_instruction='Index ' * 2000,
                  student_taxonomy_category_first_instruction='',
                  student_taxonomy_granularity_instruction='', taxonomy_plan_instruction='Plan ' * 2000)
        exec(compile(ast.Module(body=[builder], type_ignores=[]), 'prompt', 'exec'), ns)
        chunks = [{'chunk_id': i, 'text': 'x' * 1000} for i in range(40)]
        prompt = lambda group: ns['build_discovery_prompt'](('doc', 'Section'), group)
        self.assertGreater(len(prompt(chunks[:20])), coverage.REQUEST_MAX_CHARS)
        batches = list(coverage.bounded_batches(chunks, prompt_builder=prompt))
        self.assertEqual([c for b in batches for c in b], chunks)
        self.assertTrue(all(len(prompt(b)) <= coverage.REQUEST_MAX_CHARS for b in batches))
        assignment = next(n for n in ast.walk(TASK) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
        self.assertEqual(ast.unparse(assignment.value), 'build_discovery_prompt(section_key, mini_group)')

    def test_verified_batch_excludes_global_index_and_preserves_source(self):
        builder = next(n for n in ast.walk(TASK) if isinstance(n, ast.FunctionDef)
                       and n.name == 'build_discovery_prompt')
        ns = dict(chapter_ownership={1: 'index-1'},
                  chapter_structure={'chapters': [{'id': 'index-1', 'title': 'Genome structure'},
                                                  {'id': 'review', 'title': 'TO ORGANIZE'}]},
                  discovery_source_text=coverage.discovery_source_text,
                  language_instruction='Use Italian',
                  module_organization_instruction='Whole index: MLPA and genetic databases',
                  student_taxonomy_category_first_instruction='Populate all categories',
                  student_taxonomy_granularity_instruction='Preferred anchors: Mosaicism',
                  taxonomy_plan_instruction='Plan: population genetics')
        exec(compile(ast.Module(body=[builder], type_ignores=[]), 'prompt', 'exec'), ns)
        source = {'chunk_id': 1, 'page': 4, 'text': 'DNA contains coding and noncoding sequences.'}
        prompt = ns['build_discovery_prompt'](('doc', 'index-1'), [source])
        for unrelated in ('MLPA', 'genetic databases', 'Mosaicism', 'population genetics', 'Populate all'):
            self.assertNotIn(unrelated, prompt)
        self.assertIn('Genome structure', prompt)
        self.assertIn(coverage.discovery_source_text(source), prompt)
        self.assertIn('Use Italian', prompt)
        self.assertIn('only evidence for creating topics', prompt)
        review = ns['build_discovery_prompt'](('doc', 'review'), [source])
        self.assertIn('TO ORGANIZE', review)
        self.assertIn('not a subject to teach', review)
        ns['chapter_ownership'] = None
        fallback = ns['build_discovery_prompt'](('doc', 'Legacy section'), [source])
        self.assertIn('Whole index: MLPA', fallback)
        self.assertIn('not source content or a checklist', fallback)

    def test_oversized_single_complete_prompt_fails_during_planning(self):
        with self.assertRaisesRegex(ValueError, 'not discarded'):
            list(coverage.bounded_batches([{'text': 'evidence'}],
                 prompt_builder=lambda group: 'x' * (coverage.REQUEST_MAX_CHARS + 1)))

    def test_budget_scales_and_caps(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual([coverage.discovery_budget(n) for n in (1,20,40,200)], [600,600,1200,3600])
        with patch.dict(os.environ, {'DOUNO_DISCOVERY_SECONDS_PER_BATCH':'10', 'DOUNO_DISCOVERY_MIN_SECONDS':'20', 'DOUNO_DISCOVERY_MAX_SECONDS':'50'}):
            self.assertEqual([coverage.discovery_budget(n) for n in (1,3,8)], [20,30,50])
        for value in ('nan', 'inf', '-1', 'bad'):
            with patch.dict(os.environ, {'DOUNO_DISCOVERY_SECONDS_PER_BATCH':value}):
                with self.assertRaises(ValueError): coverage.discovery_budget(2)

    def test_actual_batch_plan_preserves_boundaries_and_complete_input(self):
        chunks = [{'chunk_id':i, 'document_id':i//24, 'section':str(i//12), 'text':'x'*3000} for i in range(48)]
        section_groups=defaultdict(list)
        for c in chunks: section_groups[(c['document_id'],c['section'])].append(c)
        node=next(n for n in ast.walk(TASK) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='planned_batches' for t in n.targets))
        ns={'section_groups':section_groups,'bounded_batches':coverage.bounded_batches,
            'build_discovery_prompt':lambda key, group: '\n\n'.join(coverage.discovery_source_text(c) for c in group)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'plan','exec'),ns)
        batches=[b for group in ns['planned_batches'].values() for b in group]
        self.assertEqual([c['chunk_id'] for b in batches for c in b],list(range(48)))
        for b in batches:
            self.assertEqual(len({(c['document_id'],c['section']) for c in b}),1)
            self.assertLessEqual(len(b),20)
            self.assertLessEqual(len("\n\n".join(coverage.discovery_source_text(c) for c in b)),24000)
        self.assertEqual([len(b) for b in coverage.bounded_batches([{'text':'a'}]*41)], [20,20,1])

    def test_provenance_cannot_fragment_batches_or_change_payload(self):
        chunks = [dict(chunk_id=i, document='Source.pdf', page=i+1, text='x'*900,
                       structural_provenance={'intervals': ['large metadata'*2000]}) for i in range(41)]
        batches = list(coverage.bounded_batches(chunks))
        plain = [{k:v for k,v in c.items() if k!='structural_provenance'} for c in chunks]
        self.assertEqual([len(b) for b in batches], [20,20,1])
        self.assertEqual([len(b) for b in batches], [len(b) for b in coverage.bounded_batches(plain)])
        self.assertTrue(all(a is b for a,b in zip(chunks,[c for batch in batches for c in batch])))
        for c in chunks:
            self.assertEqual(coverage.discovery_source_text(c),
                f"SOURCE: Source.pdf | PAGE: {c['page']} | SOURCE_BLOCK: {c['chunk_id']}\n{c['text']}")

    def test_batch_budget_counts_source_headers_and_separators(self):
        item = dict(text='abc',document='File',page=1,chunk_id=7)
        length = len(coverage.discovery_source_text(item))
        self.assertEqual([len(b) for b in coverage.bounded_batches([item]*2,max_chars=2*length+2)], [2])
        self.assertEqual([len(b) for b in coverage.bounded_batches([item]*2,max_chars=2*length+1)], [1,1])
        with self.assertRaises(ValueError):
            list(coverage.bounded_batches([item],max_chars=length-1))

    def test_expired_deadline_starts_no_request(self):
        create=Mock()
        with patch.object(coverage.time,'monotonic',return_value=10):
            with self.assertRaises(TimeoutError): coverage.request_json(create,'JSON',deadline=10)
        create.assert_not_called()

    def test_retry_uses_remaining_budget_and_logs_attempts(self):
        now=[0.0]; timeouts=[]
        def create(prompt):
            timeouts.append(min(240,coverage.remaining_discovery_seconds(10)))
            now[0]+=3
            if len(timeouts)==1: raise RuntimeError('transient')
            return reply()
        output=io.StringIO()
        with patch.object(coverage.time,'monotonic',side_effect=lambda:now[0]),redirect_stdout(output):
            self.assertEqual(coverage.request_json(create,'JSON',deadline=10,diagnostic_context={'batch':1}),VALID)
        self.assertEqual(timeouts,[10,7])
        self.assertEqual(output.getvalue().count('DISCOVERY_ATTEMPT'),2)
        self.assertIn('"elapsed_seconds": 3.0',output.getvalue())

    def test_late_response_and_exhausted_retry_fail(self):
        for failure in (False,True):
            now=[0]; calls=[]
            def create(prompt):
                calls.append(1); now[0]=11
                if failure: raise RuntimeError('provider timeout')
                return reply()
            with patch.object(coverage.time,'monotonic',side_effect=lambda:now[0]),redirect_stdout(io.StringIO()):
                with self.assertRaises(TimeoutError): coverage.request_json(create,'JSON',deadline=10)
            self.assertEqual(len(calls),1)

    def test_actual_request_timeout_is_clamped(self):
        node=next(n for n in ast.walk(TASK) if isinstance(n,ast.FunctionDef) and n.name=='create_candidates')
        node=__import__('copy').deepcopy(node)
        node.body=[n for n in node.body if not isinstance(n,ast.Nonlocal)]
        # Replace nonlocal counter with a global in this isolated callback harness.
        node.body.insert(0,ast.Global(names=['gpt_calls_performed']))
        ast.fix_missing_locations(node)
        call=Mock(return_value=reply())
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
        client.with_options=Mock(return_value=client)
        ns={'gpt_calls_performed':0,'discovery_deadline':10,'remaining_discovery_seconds':coverage.remaining_discovery_seconds,
            'cost':SimpleNamespace(call=call),'client':client}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'request','exec'),ns)
        with patch.object(coverage.time,'monotonic',return_value=6): ns['create_candidates']('JSON')
        self.assertEqual(call.call_args.kwargs['timeout'],4)
        client.with_options.assert_called_once_with(max_retries=0)

    def test_actual_coverage_guard_rejects_missing_chunks(self):
        node=next(n for n in ast.walk(TASK) if isinstance(n,ast.If) and ast.unparse(n.test)=='discovered_ids != eligible_ids')
        code=compile(ast.Module(body=[node],type_ignores=[]),'coverage','exec')
        with self.assertRaisesRegex(ValueError,'missing chunks'):
            exec(code,{'discovered_ids':{1},'eligible_ids':{1,2}})
        exec(code,{'discovered_ids':{1,2},'eligible_ids':{1,2}})

    def test_failed_phase_snapshot_includes_elapsed_without_double_counting(self):
        node=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='UploadPipelineLogger')
        ns={'time':time,'Optional':__import__('typing').Optional,'traceback':__import__('traceback')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'logger','exec'),ns)
        with patch.object(time,'perf_counter',return_value=100),redirect_stdout(io.StringIO()):
            logger=ns['UploadPipelineLogger']('TEST'); logger.start('TOPIC GENERATION')
        output=io.StringIO()
        with patch.object(time,'perf_counter',return_value=701),redirect_stdout(output):
            logger.failure_summary(TimeoutError('deadline'))
            self.assertEqual(logger.snapshot()['phase_totals']['TOPIC GENERATION'],601)
            logger.end('TOPIC GENERATION')
            self.assertEqual(logger.snapshot()['phase_totals']['TOPIC GENERATION'],601)
        self.assertIn('601.00 s',output.getvalue())
