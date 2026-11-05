"""No provider or database calls: fixtures model the current SDK response shapes."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import ai_cost_test as cost
from chat_stream import stream_chat

spec = importlib.util.spec_from_file_location('cost_report', Path(__file__).resolve().parents[1] / 'scripts/report_ai_cost_test.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
PRICES = json.loads(report.DEFAULT_PRICES.read_text())


def response(model='gpt-4o-mini'):
    return NS(model=model, usage=NS(prompt_tokens=1000, completion_tokens=200, total_tokens=1200,
        prompt_tokens_details=NS(cached_tokens=100), completion_tokens_details=NS(reasoning_tokens=50)), choices=[])


class Stream:
    def __init__(self, interrupted=False):
        self.interrupted, self.closed = interrupted, False
    def __iter__(self):
        yield NS(choices=[NS(delta=NS(content='Hello'), finish_reason=None)])
        if self.interrupted:
            raise TimeoutError('SECRET output')
        yield NS(choices=[NS(delta=NS(content=' world'), finish_reason='stop')])
        yield response()
    def close(self):
        self.closed = True


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.env = patch.dict(os.environ, {'DOUNO_AI_COST_TEST':'1', 'DOUNO_AI_COST_TEST_ID':'mock-test', 'DOUNO_AI_COST_TEST_OUTPUT_DIR':self.temp.name})
        self.env.start()
        cost._failed_writes = 0
        cost._warned = False
    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()
    def events(self):
        return report.load_events([Path(self.temp.name)])[0]
    def call(self, fn=lambda **kw: response(), **tags):
        return cost.call(fn, cost_feature='quiz', cost_suboperation='questions', model='gpt-4o-mini', **tags)
    def test_disabled_exact_arguments_no_files(self):
        with patch.dict(os.environ, {'DOUNO_AI_COST_TEST':'0'}):
            seen = []
            original = object()
            result = self.call(lambda **kw: (seen.append(kw), original)[1], stream=True)
            self.assertIs(result, original)
            self.assertEqual(seen, [{'model':'gpt-4o-mini', 'stream':True}])
            self.assertEqual(list(Path(self.temp.name).iterdir()), [])
    def test_usage_before_application_failure(self):
        @cost.operation('quiz')
        def run():
            self.call()
            raise ValueError('private answer')
        with self.assertRaises(ValueError): run()
        calls = [e for e in self.events() if e['event']=='call_end']
        self.assertEqual(calls[0]['usage']['input_tokens'],1000)
        self.assertEqual(calls[0]['status'],'succeeded')
        self.assertEqual(self.events()[-1]['status'],'failed')
        self.assertNotIn('private answer',json.dumps(self.events()))
    def test_provider_failure_unknown(self):
        def provider(**kw): raise TimeoutError('private token')
        with self.assertRaises(TimeoutError): self.call(provider)
        end = self.events()[-1]
        self.assertEqual(end['status'],'failed')
        self.assertIsNone(report.price_call(end,PRICES)[0])
    def test_missing_usage_and_speech_unknown(self):
        for value in [NS(), NS(content=b'private MP3')]:
            self.call(lambda **kw:value)
        summaries,_,_=report.build_report(self.events(),PRICES)
        self.assertEqual(summaries[0]['unknown_cost_calls'],2)
        self.assertFalse(summaries[0]['full_observed_cost_available'])
    def test_embeddings_actual_usage(self):
        result = NS(model='text-embedding-3-small',usage=NS(prompt_tokens=1000,total_tokens=1000))
        cost.call(lambda **kw:result,cost_feature='ingestion',cost_suboperation='embedding',model=result.model)
        price,_=report.price_call(self.events()[-1],PRICES)
        self.assertEqual(str(price),'0.00002')
    def test_decimal_cached_and_reasoning_not_double_counted(self):
        self.call()
        price,_=report.price_call(self.events()[-1],PRICES)
        self.assertEqual(str(price),'0.0002625')
    def test_unknown_model_and_missing_cached_usage(self):
        for e in [dict(model='unknown',usage={}),dict(model='gpt-4o-mini',usage={'input_tokens':1,'output_tokens':1})]:
            self.assertIsNone(report.price_call(e,PRICES)[0])
    def test_stream_output_unchanged_and_terminal_usage(self):
        async def collect(active):
            with patch.dict(os.environ, {'DOUNO_AI_COST_TEST':active}):
                upstream=Stream()
                output = stream_chat(lambda: self.call(lambda **kw:upstream,stream=True))
                items=[json.loads(x) async for x in output.body_iterator]
                self.assertTrue(upstream.closed)
                return items
        self.assertEqual(asyncio.run(collect('0')),asyncio.run(collect('1')))
        ends=[e for e in self.events() if e['event']=='call_end']
        self.assertEqual(len(ends),1)
        self.assertEqual(ends[0]['usage']['output_tokens'],200)
    def test_interrupted_stream_usage_unknown(self):
        async def run():
            upstream=Stream(True)
            output=stream_chat(lambda:self.call(lambda **kw:upstream,stream=True))
            return [json.loads(x) async for x in output.body_iterator]
        self.assertEqual(asyncio.run(run())[-1]['type'],'error')
        self.assertEqual([e for e in self.events() if e['event']=='call_end'][0]['usage'],{})
    def test_streaming_operation_context_survives_route_return(self):
        @cost.operation('ask')
        def route():return stream_chat(lambda:self.call(lambda **kw:Stream(),stream=True))
        async def run():return [x async for x in route().body_iterator]
        asyncio.run(run())
        ev=self.events();start=next(e for e in ev if e['event']=='operation_start')
        end=next(e for e in ev if e['event']=='call_end')
        self.assertEqual(start['operation_id'],end['operation_id'])
        self.assertEqual(ev[-1]['event'],'operation_end')
    def test_background_and_distinct_coverage_scopes(self):
        @cost.operation('taxonomy')
        def taxonomy(available, selected):
            cost.metrics(scope_kind='module',teaching_available=available,teaching_selected=selected,selected_rows=selected,selection_limit=400,topics_generated=20)
            self.call()
            return True
        @cost.operation('ingestion')
        def upload():return cost.background(lambda:(taxonomy(800,400),taxonomy(100,100)))
        task=upload()
        before,_,_=report.build_report(self.events(),PRICES)
        self.assertEqual(before[0]['operation_status'],'incomplete')
        task()
        summary,_,markdown=report.build_report(self.events(),PRICES)
        scopes=json.loads(summary[0]['taxonomy_scopes'])
        self.assertEqual([x['input_coverage_percent'] for x in scopes],[50,100])
        self.assertIn('WARNING',markdown)
        self.assertEqual(summary[0]['operation_status'],'observed_components_completed')
    def test_session_aggregation_refills_repairs_planner(self):
        @cost.operation('quiz')
        def quiz():
            cost.metrics(requested_count=15,difficulty='hard',style='balanced')
            self.call(cost_phase='initial')
            self.call(cost_phase='refill',cost_attempt=1)
            self.call(cost_phase='repair',cost_attempt=2)
            cost.metrics(saved_count=15)
        @cost.operation('planner')
        def planner():self.call()
        quiz();planner()
        summary,requests,_=report.build_report(self.events(),PRICES,feature='study_session')
        self.assertEqual(summary[0]['observed_openai_calls'],4)
        self.assertEqual(summary[0]['feature'],'study_session')
        self.assertEqual(requests[2]['phase'],'repair')
    def test_logging_failure_fail_open(self):
        original=response()
        with patch.object(Path,'mkdir',side_effect=OSError('SECRET')):
            self.assertIs(self.call(lambda **kw:original),original)
        self.call()
        self.assertGreater(self.events()[-1]['writer_failures'],0)
    def test_metadata_allowlist(self):
        cost.metrics(filename='private.pdf',prompt='secret',source_pages=100,document_format='pdf',difficulty='private')
        payload=json.dumps(self.events())
        self.assertNotIn('private',payload);self.assertNotIn('secret',payload)
        self.assertEqual(self.events()[-1]['metadata']['source_pages'],100)
    def test_merge_deduplicates_and_warns_truncated_data(self):
        self.call()
        source=next(Path(self.temp.name).rglob('*.jsonl'))
        (source.parent/'copy.jsonl').write_text(source.read_text()+'{broken\n')
        events,warnings=report.load_events([Path(self.temp.name)])
        self.assertEqual(len(events),2);self.assertTrue(warnings)
    def test_concurrency_keeps_operation_ids(self):
        @cost.operation('quiz')
        async def run():
            await asyncio.to_thread(self.call)
        async def all_runs():await asyncio.gather(run(),run())
        asyncio.run(all_runs())
        ids=[e['operation_id'] for e in self.events() if e['event']=='call_end']
        self.assertEqual(len(set(ids)),2)


    def test_generator_close_marks_interrupted_and_resets_context(self):
        @cost.operation('ingestion')
        def generate():
            self.call()
            yield 'first'
            yield 'second'
        source=generate()
        self.assertEqual(next(source),'first')
        source.close()
        self.assertIsNone(cost._context.get())
        self.assertEqual(self.events()[-1]['status'],'interrupted')
    def test_exact_limit_does_not_claim_exclusion(self):
        @cost.operation('taxonomy')
        def run():
            cost.metrics(scope_kind='document',teaching_available=400,selected_rows=400,teaching_selected=400,selection_limit=400)
        run()
        summaries,_,text=report.build_report(self.events(),PRICES)
        self.assertFalse(summaries[0]['taxonomy_limited'])
        self.assertNotIn('WARNING',text)
        self.assertTrue(json.loads(summaries[0]['taxonomy_scopes'])[0]['limit_reached'])
    def test_nonempty_filter_is_not_misattributed_to_cap(self):
        @cost.operation('taxonomy')
        def run():cost.metrics(scope_kind='document',teaching_available=100,selected_rows=100,teaching_selected=90,selection_limit=400)
        run()
        summaries,_,text=report.build_report(self.events(),PRICES)
        self.assertTrue(summaries[0]['taxonomy_limited'])
        self.assertEqual(json.loads(summaries[0]['taxonomy_scopes'])[0]['excluded_by_limit'],0)
        self.assertIn('WARNING',text)
    def test_pricing_rejects_invalid_cache_and_unknown_tier(self):
        self.call()
        e=self.events()[-1]
        e['usage']['cached_input_tokens']=1001
        self.assertIsNone(report.price_call(e,PRICES)[0])
        e['usage']['cached_input_tokens']=0;e['service_tier']='priority'
        self.assertIsNone(report.price_call(e,PRICES)[0])
    def test_stream_usage_option_only_enabled(self):
        seen=[]
        upstream=Stream()
        source=self.call(lambda **kw:(seen.append(kw),upstream)[1],stream=True)
        list(source);source.close()
        self.assertEqual(seen[0]['stream_options'],{'include_usage':True})
    def test_cli_writes_three_reports_and_pricing_snapshot(self):
        import subprocess,sys
        self.call()
        output=Path(self.temp.name)/'report'
        result=subprocess.run([sys.executable,str(report.DEFAULT_PRICES.parents[1]/'scripts/report_ai_cost_test.py'),self.temp.name,'--out',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for name in ['summary.csv','requests.csv','summary.md','pricing-used.json']:
            self.assertTrue((output/name).exists())
        self.assertIn('not provider invoices',(output/'summary.md').read_text())


if __name__=='__main__':unittest.main()
