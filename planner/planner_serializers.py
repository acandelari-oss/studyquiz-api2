"""Lightweight serializers for Planner domain objects.

The Planner domain remains dataclass-based and framework-independent. These
helpers convert domain objects into JSON-compatible dictionaries for temporary
API responses.
"""

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


def serialize_planner_domain(value: Any) -> Any:
    """Convert Planner domain values into JSON-compatible primitives."""

    if is_dataclass(value):
        return {
            field.name: serialize_planner_domain(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            serialize_planner_domain(key): serialize_planner_domain(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize_planner_domain(item) for item in value]

    return value
