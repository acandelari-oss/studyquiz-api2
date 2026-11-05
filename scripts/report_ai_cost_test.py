#!/usr/bin/env python3
"""Offline metadata-only benchmark report. Never imports main or calls an API/DB."""
import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

DEFAULT_PRICES = Path(__file__).resolve().parents[1] / 'benchmark' / 'ai_prices.json'


def price_call(call, config):
    model = call.get('returned_model') or call.get('model')
    price = next((v for k, v in config['models'].items() if model == k or model in v.get('snapshots', [])), None)
    if price is None:
        return None, 'unknown_model'
    if call.get('service_tier') not in (None, 'default', 'standard', 'auto'):
        return None, 'unknown_service_tier'
    u = call.get('usage') or {}
    def count(key):
        value = u.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(key)
        return value
    try:
        inp = count('input_tokens')
        if price.get('kind') == 'embedding':
            amount = Decimal(inp) * Decimal(price['input'])
        elif price.get('kind') == 'speech':
            # Binary MP3 does not provide these; do not estimate from bytes.
            amount = Decimal(inp) * Decimal(price['input']) + Decimal(count('audio_output_tokens')) * Decimal(price['audio_output'])
        else:
            out, cached = count('output_tokens'), count('cached_input_tokens')
            if cached > inp:
                return None, 'invalid_usage'
            if u.get('audio_input_tokens', 0) or u.get('audio_output_tokens', 0):
                return None, 'unpriced_audio_category'
            amount = Decimal(inp-cached)*Decimal(price['input']) + Decimal(cached)*Decimal(price['cached_input']) + Decimal(out)*Decimal(price['output'])
        # Reasoning tokens are already a subset of output, never an extra charge.
        return amount / Decimal(1000000), 'priced'
    except (KeyError, ValueError, TypeError, InvalidOperation):
        return None, 'missing_or_invalid_usage'


def load_events(paths):
    events, seen, warnings = [], set(), []
    files = sorted({f for path in paths for f in (path.rglob('*.jsonl') if path.is_dir() else [path])})
    for path in files:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            warnings.append('An input file could not be read.')
            continue
        for number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get('event_id'), str) or event.get('schema_version') != 1:
                    raise ValueError()
                if event['event_id'] in seen:
                    continue
                seen.add(event['event_id'])
                events.append(event)
            except (ValueError, TypeError):
                warnings.append('Malformed/incomplete diagnostic line; totals may omit records.')
    return sorted(events, key=lambda e: e.get('timestamp', '')), sorted(set(warnings))


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ['test_id']
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(events, config, warnings=(), feature=None):
    summaries, requests, sections = [], [], []
    for test in sorted({e['test_id'] for e in events}):
        records = [e for e in events if e['test_id'] == test]
        calls, starts, ends, metadata, jobs = {}, {}, {}, {}, {}
        for e in records:
            op = e.get('operation_id')
            if e['event'] == 'operation_start':
                starts[op] = e
            elif e['event'] == 'operation_end':
                ends[op] = e
            elif e['event'] == 'metrics':
                metadata.setdefault(op, []).append(e.get('metadata', {}))
            elif e['event'] in ('call_start', 'call_end'):
                old = calls.get(e['call_id'])
                if not old or e['event'] == 'call_end':
                    calls[e['call_id']] = e
            elif e['event'].startswith('background_'):
                jobs[e['job_id']] = e['event']
        unfinished = len(set(starts)-set(ends)) + sum(v != 'background_end' for v in jobs.values())
        statuses = [e['status'] for e in ends.values()]
        outcomes = [m['outcome'] for values in metadata.values() for m in values if 'outcome' in m]
        if 'failed' in statuses + outcomes:
            status = 'failed'
        elif unfinished or 'interrupted' in statuses + outcomes:
            status = 'incomplete'
        elif 'fallback' in outcomes:
            status = 'completed_with_fallback'
        elif starts:
            status = 'observed_components_completed'
        else:
            status = 'unknown'
        total, unknown, missing_usage, inputs, outputs = Decimal(0), 0, 0, 0, 0
        for e in calls.values():
            amount, pricing_status = price_call(e, config)
            if amount is None:
                unknown += 1
            else:
                total += amount
            u = e.get('usage') or {}
            missing_usage += not bool(u)
            inputs += u.get('input_tokens') or 0
            outputs += u.get('output_tokens') or 0
            op = e.get('operation_id')
            op_outcome = next((m['outcome'] for m in reversed(metadata.get(op, [])) if 'outcome' in m), None)
            requests.append(dict(test_id=test, call_id=e['call_id'], operation_id=op,
                feature=e.get('call_feature'), component=starts.get(op, {}).get('feature', e.get('feature')),
                suboperation=e.get('suboperation'), model=e.get('model'), returned_model=e.get('returned_model'),
                timestamp=e.get('timestamp'), provider_status=e.get('status', 'unfinished'),
                application_outcome=e.get('application_outcome', 'unknown'),
                operation_outcome=op_outcome or ends.get(op, {}).get('status', 'unknown'),
                input_tokens=u.get('input_tokens'), output_tokens=u.get('output_tokens'),
                cached_input_tokens=u.get('cached_input_tokens'), other_usage=json.dumps(u, sort_keys=True),
                calculated_cost_usd=str(amount) if amount is not None else '', pricing_status=pricing_status,
                pricing_version=config['version'], duration_ms=e.get('duration_ms'),
                application_attempt=e.get('metadata', {}).get('application_attempt'),
                phase=e.get('metadata', {}).get('phase'), sdk_retry_visibility='unobserved'))
        configs, coverage, documents, counts = [], [], {}, []
        for op, values in metadata.items():
            merged = {k: v for m in values for k, v in m.items() if k != 'outcome'}
            if 'teaching_available' in merged:
                available, selected, selected_rows = (merged.get(k) for k in ('teaching_available', 'teaching_selected', 'selected_rows'))
                limit = merged.get('selection_limit')
                scope = dict(operation_id=op, scope_kind=merged.get('scope_kind'), teaching_available=available,
                    teaching_selected=selected, selected_rows=selected_rows, selection_limit=limit,
                    limit_reached=selected_rows >= limit if selected_rows is not None and limit else None,
                    excluded_by_limit=max(0, available-selected_rows) if available is not None and selected_rows is not None else None,
                    input_coverage_percent=round(100*selected/available, 2) if available and selected is not None else None,
                    topics_generated=merged.get('topics_generated'), status=ends.get(op, {}).get('status', 'unfinished'))
                coverage.append(scope)
            for m in values:
                if 'document_index' in m:
                    key = (op, m['document_index'])
                    documents.setdefault(key, dict(operation_id=op)).update(m)
            cfg = {k: merged[k] for k in ('difficulty', 'style', 'requested_count') if k in merged}
            if cfg:
                configs.append(dict(operation_id=op, **cfg))
            cnt = {k: merged[k] for k in ('generated_count', 'saved_count') if k in merged}
            if cnt:
                counts.append(dict(operation_id=op, **cnt))
        limited = any(scope['teaching_available'] is not None and scope['teaching_selected'] is not None and scope['teaching_selected'] < scope['teaching_available'] for scope in coverage)
        writer_failures = max((r.get('writer_failures', 0) for r in records), default=0)
        incomplete = bool(unknown or unfinished or warnings or writer_failures)
        duration = (datetime.fromisoformat(records[-1]['timestamp']) - datetime.fromisoformat(records[0]['timestamp'])).total_seconds()
        successful_pdf_pages = sum(d.get('source_pages') or 0 for d in documents.values() if d.get('document_format') == 'pdf' and 'chunks_generated' in d)
        # Only meaningful when all observed ingestion components succeeded and coverage is complete.
        per100 = str(total * 100 / successful_pdf_pages) if successful_pdf_pages and not incomplete and not limited and status == 'observed_components_completed' and all(d.get('document_format') == 'pdf' for d in documents.values()) and all(e.get('call_feature') in ('ingestion', 'taxonomy') for e in calls.values()) else ''
        summary = dict(test_id=test, date=records[0]['timestamp'][:10], feature=feature or ', '.join(sorted({e.get('call_feature', 'unknown') for e in calls.values()})),
            test_configuration=json.dumps(configs), operation_status=status, observed_openai_calls=len(calls),
            input_tokens=inputs, output_tokens=outputs, known_calculated_usd=str(total), full_observed_cost_available=not incomplete,
            unknown_cost_calls=unknown, missing_usage_calls=missing_usage, unfinished_operations=unfinished,
            application_extra_calls=sum(e.get('metadata', {}).get('phase') in ('refill', 'repair') or (e.get('metadata', {}).get('application_attempt') or 0) > 1 for e in calls.values()),
            duration_seconds=round(duration, 3), output_counts=json.dumps(counts), documents=json.dumps(list(documents.values())),
            taxonomy_scopes=json.dumps(coverage), taxonomy_limited=limited, calculated_usd_per_100_completed_pdf_pages=per100,
            writer_failures=writer_failures, pricing_version=config['version'], sdk_retry_visibility='unobserved')
        summaries.append(summary)
        text = [f'## {test}', '', f'**Known calculated AI cost: ${total:.8f} USD**',
            'Full observed operation cost: unavailable/incomplete.' if incomplete else 'All observed calls are priced. Hidden SDK attempts are not measured.',
            '', f'- Feature: {summary["feature"]}', f'- Outcome: {status.replace("_", " ")}', f'- Observed OpenAI invocations: {len(calls)}',
            f'- Reported input/output tokens: {inputs:,} / {outputs:,}', f'- Unknown-cost calls: {unknown}; unfinished operations: {unfinished}',
            f'- Observed elapsed time: {duration:.2f} seconds', f'- Configuration: {"; ".join(", ".join(f"{k.replace(chr(95), chr(32))}: {v}" for k, v in cfg.items() if k != "operation_id") for cfg in configs) or "not recorded"}', f'- Output counts: {"; ".join(", ".join(f"{k.replace(chr(95), chr(32))}: {v}" for k, v in count.items() if k != "operation_id") for count in counts) or "not recorded"}', f'- Observed additional application calls: {summary["application_extra_calls"]}']
        components = {}
        for request in requests:
            if request['test_id'] != test:
                continue
            key = request['component'] or request['feature']
            component = components.setdefault(key, [0, Decimal(0), 0])
            component[0] += 1
            if request['calculated_cost_usd'] == '':
                component[2] += 1
            else:
                component[1] += Decimal(request['calculated_cost_usd'])
        if components:
            text += ['', '### Activity components', '', '| Component | Calls | Known USD subtotal | Unknown costs |', '|---|---|---|---|']
            for key, (number, subtotal, unknown_count) in components.items():
                text.append(f'| {key} | {number} | ${subtotal:.8f} | {unknown_count} |')
        if documents:
            text += ['', '### Documents', '', '| Document index | Format | Source pages | Extracted pages | Generated chunks |', '|---|---|---|---|---|']
            for d in documents.values():
                text.append('| '+' | '.join(str(d.get(k)) if d.get(k) is not None else 'unavailable' for k in ['document_index','document_format','source_pages','extracted_pages','chunks_generated'])+' |')
        if coverage:
            text += ['', '### Taxonomy input coverage — each processing scope separately', '', '| Scope | Available teaching chunks | Selected input | Limit reached | Excluded by limit | Coverage | Topics | Status |', '|---|---|---|---|---|---|---|---|']
            for i, c in enumerate(coverage, 1):
                text.append(f'| {i} ({c["scope_kind"]}) | {c["teaching_available"]} | {c["teaching_selected"]} | {c["limit_reached"]} | {c["excluded_by_limit"]} | {c["input_coverage_percent"]}% | {c["topics_generated"]} | {c["status"]} |')
            if limited:
                text += ['', '**WARNING: taxonomy input was limited relative to available teaching material. Low cost does not demonstrate complete source coverage.**']
        if per100:
            text += ['', f'Calculated AI cost per 100 completed PDF pages: ${Decimal(per100):.8f}.']
        elif documents:
            text += ['', 'Cost per 100 fully processed pages: unavailable (check coverage, completion, formats and missing usage).']
        sections.append('\n'.join(text))
    preamble = '# DOUNO AI cost benchmark\n\nCalculated from provider-reported usage; not provider invoices. No tokens are inferred from content length.\n\nSDK-internal retries are unobserved. Scenario grouping assumes one controlled test at a time. Component completion does not certify that a student finished a whole Study Session.\n\n'
    if warnings:
        preamble += '**Input warnings:** '+ ' '.join(warnings)+'\n\n'
    return summaries, requests, preamble + '\n\n'.join(sections) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='JSONL files or directories (merged, deduplicated)')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prices', default=DEFAULT_PRICES, type=Path)
    parser.add_argument('--feature', help='Scenario label, e.g. study_session; does not override observed outcomes')
    args = parser.parse_args()
    events, warnings = load_events(args.inputs)
    if not events:
        parser.error('No readable benchmark events found; no zero-cost report produced.')
    config = json.loads(args.prices.read_text())
    summaries, requests, markdown = build_report(events, config, warnings, args.feature)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / '.gitignore').write_text('*\n')
    _write_csv(args.out / 'summary.csv', summaries)
    _write_csv(args.out / 'requests.csv', requests)
    (args.out / 'summary.md').write_text(markdown)
    (args.out / 'pricing-used.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Reports written to {args.out.resolve()}')


if __name__ == '__main__':
    main()
