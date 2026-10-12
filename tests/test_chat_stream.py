import asyncio
import json
import unittest
from types import SimpleNamespace as NS
from chat_stream import stream_chat

class Upstream:
    def __init__(self, pieces, finish='stop'):
        self.pieces, self.finish, self.closed = pieces, finish, False
    def __iter__(self):
        for text in self.pieces:
            yield NS(choices=[NS(delta=NS(content=text), finish_reason=None)])
        yield NS(choices=[NS(delta=NS(content=None), finish_reason=self.finish)])
    def close(self):
        self.closed = True

class StreamingTests(unittest.TestCase):
    def collect(self, source, **kwargs):
        async def run():
            return [json.loads(line) async for line in stream_chat(lambda: source, **kwargs).body_iterator]
        return asyncio.run(run())
    def test_text_metadata_and_close(self):
        source = Upstream(['Hello', ' world'])
        events = self.collect(source, metadata={'sources': ['book']})
        self.assertEqual([e['text'] for e in events[:-1]], ['Hello', 'Hello world'])
        self.assertEqual(events[-1]['data']['sources'], ['book'])
        self.assertTrue(source.closed)
    def test_partial_json_never_exposes_scoring_or_syntax(self):
        content = json.dumps({'evaluation':'correct','score':1,'feedback':'Good "idea" ☕','explanation':'Details','hint':'Try'})
        source = Upstream(list(content))
        events = self.collect(source, structured=True)
        self.assertEqual(events[-1]['data']['score'], 1)
        texts = [e['text'] for e in events if e['type']=='text']
        self.assertIn('Good', texts)
        self.assertNotIn('evaluation', ''.join(texts))
        self.assertTrue(source.closed)
    def test_incomplete_response_is_error(self):
        source = Upstream(['partial'], 'length')
        self.assertEqual(self.collect(source)[-1]['type'], 'error')
        self.assertTrue(source.closed)
