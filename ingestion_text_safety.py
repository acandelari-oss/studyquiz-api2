"""Narrow PostgreSQL text compatibility; no linguistic normalization."""
import json


def strip_nul(value):
    return value.replace('\x00', '') if isinstance(value, str) else value


def sanitize_acquired_text(document):
    """Clean study text before structure detection/chunking/embeddings.

    Keep digital_text and its offsets as original extraction evidence.
    Diagnostics contain counts and locations only, never document contents.
    """
    for block in document.blocks:
        counts = {}
        for field in ('text', 'raw_text', 'section'):
            value = getattr(block, field, None)
            if isinstance(value, str) and '\x00' in value:
                counts[field] = value.count('\x00')
                setattr(block, field, strip_nul(value))
        if counts:
            print('INGESTION_NUL_REMOVED', json.dumps({
                'page': block.page, 'block': block.block_index, 'fields': counts}))


def safe_ingestion_row(row, json_fields=()):
    """Final guard for text parameters and explicitly identified JSON payloads."""
    def clean(value):
        if isinstance(value, str):
            return strip_nul(value)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value
    result = {key: strip_nul(value) for key, value in row.items()}
    for key in json_fields:
        if result.get(key) is not None:
            result[key] = json.dumps(clean(json.loads(result[key])), ensure_ascii=False)
    return result
