"""Opt-in local metadata benchmark. No provider retries, DB access or content logging."""
import contextvars
import functools
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

_context = contextvars.ContextVar('ai_cost_test', default=None)
_lock = threading.Lock()
_failed_writes = 0
_warned = False
_process = uuid.uuid4().hex[:8]
# Never accept arbitrary diagnostic payloads (not even exception messages).
_NUMBERS = {'requested_count', 'generated_count', 'saved_count', 'application_attempt',
            'document_index', 'source_pages', 'extracted_pages', 'slides', 'chunks_generated',
            'teaching_available', 'teaching_selected', 'selected_rows', 'selection_limit',
            'topics_generated', 'duration_ms'}
_ENUMS = {
    'difficulty': {'easy', 'medium', 'hard'},
    'style': {'exam', 'balanced', 'reasoning'},
    'phase': {'initial', 'refill', 'repair'},
    'document_format': {'pdf', 'docx', 'pptx'},
    'scope_kind': {'module', 'document', 'project'},
    'outcome': {'completed', 'failed', 'fallback', 'unknown', 'interrupted', 'scheduled'},
}


def enabled():
    return os.getenv('DOUNO_AI_COST_TEST') == '1'


def _safe_label(value):
    return value if isinstance(value, str) and re.fullmatch(r'[a-zA-Z0-9_.:-]{1,100}', value) else 'unknown'


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _number(value):
    return value if type(value) is int and value >= 0 else None


def _metadata(values):
    result = {}
    for key, value in values.items():
        if key in _NUMBERS:
            result[key] = _number(value)  # null explicitly means unavailable
        elif key in _ENUMS:
            result[key] = value if isinstance(value, str) and value in _ENUMS[key] else 'unknown'
    return result


def _emit(event, **fields):
    global _failed_writes, _warned
    if not enabled():
        return
    try:
        test_id = os.getenv('DOUNO_AI_COST_TEST_ID', 'local-test')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', test_id):
            raise ValueError('invalid test identifier')
        context = _context.get() or {}
        data = dict(schema_version=1, event_id=uuid.uuid4().hex, event=event,
                    timestamp=datetime.now(timezone.utc).isoformat(), test_id=test_id,
                    operation_id=context.get('id', test_id), feature=context.get('feature', 'unscoped'),
                    writer_failures=_failed_writes, **fields)
        directory = Path(os.getenv('DOUNO_AI_COST_TEST_OUTPUT_DIR', 'benchmark-output')) / test_id
        line = json.dumps(data, ensure_ascii=True) + '\n'
        with _lock:
            directory.mkdir(parents=True, exist_ok=True)
            # Keep arbitrary configured output directories out of Git too.
            ignore = directory / '.gitignore'
            if not ignore.exists():
                ignore.write_text('*\n')
            with (directory / f'backend-{os.getpid()}-{_process}.jsonl').open('a') as file:
                file.write(line)
    except Exception:
        _failed_writes += 1
        if not _warned:
            _warned = True
            try:
                print('AI cost benchmark: diagnostic write failed; report may be incomplete.', file=sys.stderr)
            except Exception:
                pass


def metrics(**values):
    if enabled():
        _emit('metrics', metadata=_metadata(values))


def outcome(value):
    if enabled():
        metrics(outcome=value)


def _start(feature):
    parent = _context.get()
    ctx = {'id': uuid.uuid4().hex, 'feature': feature}
    token = _context.set(ctx)
    _emit('operation_start', parent_id=parent['id'] if parent else None)
    return ctx, token, time.monotonic()


def _finish(started, status='completed'):
    _emit('operation_end', status=status, duration_ms=round((time.monotonic()-started)*1000))


def operation(feature):
    """Keep original signature/DI; no request payload introspection."""
    def decorate(fn):
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def generator(*args, **kwargs):
                if not enabled():
                    return fn(*args, **kwargs)
                # Set/reset in each next() context: Starlette may use different workers.
                ctx, token, started = _start(feature)
                _context.reset(token)
                def events():
                    source = fn(*args, **kwargs)
                    status = 'interrupted'
                    try:
                        while True:
                            token = _context.set(ctx)
                            try:
                                item = next(source)
                            except StopIteration:
                                status = 'completed'
                                break
                            except BaseException:
                                status = 'failed'
                                raise
                            finally:
                                _context.reset(token)
                            yield item
                    finally:
                        token = _context.set(ctx)
                        try:
                            source.close()
                        finally:
                            _finish(started, status)
                            _context.reset(token)
                return events()
            return generator

        def finish_result(result, ctx, started):
            if hasattr(result, 'body_iterator'):
                source = result.body_iterator
                async def events():
                    status = 'interrupted'
                    try:
                        while True:
                            token = _context.set(ctx)
                            try:
                                item = await source.__anext__()
                            except StopAsyncIteration:
                                status = 'completed'
                                break
                            finally:
                                _context.reset(token)
                            yield item
                    finally:
                        token = _context.set(ctx)
                        try:
                            close = getattr(source, 'aclose', None)
                            if close is not None:
                                await close()
                        finally:
                            _finish(started, status)
                            _context.reset(token)
                result.body_iterator = events()
            else:
                _finish(started, 'failed' if result is False else 'completed')
            return result

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def asynchronous(*args, **kwargs):
                if not enabled():
                    return await fn(*args, **kwargs)
                ctx, token, started = _start(feature)
                try:
                    return finish_result(await fn(*args, **kwargs), ctx, started)
                except BaseException:
                    _finish(started, 'failed')
                    raise
                finally:
                    _context.reset(token)
            return asynchronous
        @functools.wraps(fn)
        def synchronous(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            ctx, token, started = _start(feature)
            try:
                return finish_result(fn(*args, **kwargs), ctx, started)
            except BaseException:
                _finish(started, 'failed')
                raise
            finally:
                _context.reset(token)
        return synchronous
    return decorate


def background(fn):
    """Bind parent explicitly; record scheduled work even if it never starts."""
    if not enabled():
        return fn
    parent = _context.get()
    job_id = uuid.uuid4().hex
    _emit('background_scheduled', job_id=job_id)
    @functools.wraps(fn)
    def run(*args, **kwargs):
        token = _context.set(parent)
        _emit('background_start', job_id=job_id)
        status = 'failed'
        try:
            result = fn(*args, **kwargs)
            status = 'completed'
            return result
        finally:
            _emit('background_end', job_id=job_id, status=status)
            _context.reset(token)
    return run


def usage_of(response):
    usage = _get(response, 'usage')
    if usage is None:
        return {}
    fields = {'input_tokens': _get(usage, 'prompt_tokens', _get(usage, 'input_tokens')),
              'output_tokens': _get(usage, 'completion_tokens', _get(usage, 'output_tokens')),
              'total_tokens': _get(usage, 'total_tokens')}
    inp = _get(usage, 'prompt_tokens_details', _get(usage, 'input_tokens_details'))
    out = _get(usage, 'completion_tokens_details', _get(usage, 'output_tokens_details'))
    for target, source, key in [('cached_input_tokens', inp, 'cached_tokens'),
                               ('audio_input_tokens', inp, 'audio_tokens'),
                               ('audio_output_tokens', out, 'audio_tokens'),
                               ('reasoning_tokens', out, 'reasoning_tokens'),
                               ('accepted_prediction_tokens', out, 'accepted_prediction_tokens'),
                               ('rejected_prediction_tokens', out, 'rejected_prediction_tokens')]:
        fields[target] = _get(source, key)
    return {key: number for key, value in fields.items() if (number := _number(value)) is not None}


class _Stream:
    def __init__(self, source, complete):
        self.source, self.complete = source, complete
        self.usage, self.model, self.status, self.closed = {}, None, 'interrupted', False
        self.tier = None
    def __iter__(self):
        try:
            for chunk in self.source:
                try:
                    self.usage = usage_of(chunk) or self.usage
                    self.model = _get(chunk, 'model') or self.model
                    self.tier = _get(chunk, 'service_tier') or self.tier
                except Exception:
                    pass
                yield chunk
            self.status = 'succeeded'
        except BaseException:
            self.status = 'interrupted'
            raise
        finally:
            self.complete(self.status, self.usage, self.model, self.tier)
    def close(self):
        try:
            self.source.close()
        finally:
            if not self.closed:
                self.closed = True
                self.complete(self.status, self.usage, self.model, self.tier)


def call(fn, *, cost_feature, cost_suboperation, cost_attempt=None, cost_phase=None, **kwargs):
    if not enabled():
        return fn(**kwargs)
    call_id = uuid.uuid4().hex
    started = time.monotonic()
    ctx = _context.get()
    model = _safe_label(kwargs.get('model'))
    metadata = _metadata({'application_attempt': cost_attempt, 'phase': cost_phase})
    base = dict(call_id=call_id, call_feature=cost_feature, suboperation=cost_suboperation,
                model=model, metadata=metadata, sdk_retry_visibility='unobserved')
    _emit('call_start', **base)
    done = False
    def complete(status, usage=None, returned_model=None, service_tier=None):
        nonlocal done
        if done:
            return
        done = True
        token = _context.set(ctx)
        try:
            _emit('call_end', **base, status=status, usage=usage or {},
                  returned_model=_safe_label(returned_model) if returned_model else None,
                  service_tier=_safe_label(service_tier) if service_tier else None,
                  application_outcome='unknown', duration_ms=round((time.monotonic()-started)*1000))
        finally:
            _context.reset(token)
    try:
        if kwargs.get('stream'):
            kwargs = {**kwargs, 'stream_options': {**kwargs.get('stream_options', {}), 'include_usage': True}}
        result = fn(**kwargs)
    except BaseException:
        complete('failed')
        raise
    if kwargs.get('stream'):
        return _Stream(result, complete)
    try:
        complete('succeeded', usage_of(result), _get(result, 'model'), _get(result, 'service_tier'))
    except Exception:
        # Provider response must survive unexpected SDK metadata shapes.
        complete('succeeded')
    return result


def bind_call(fn, **tags):
    return functools.partial(call, fn, **tags) if enabled() else fn
