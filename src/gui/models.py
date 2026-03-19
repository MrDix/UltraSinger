"""Data models for the UltraSinger GUI."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class QueueItem:
    """A single item in the conversion queue."""

    input_source: str  # URL or file path
    input_type: str  # "url" or "file"
    title: str  # display title (from browser or filename)
    status: str = "pending"  # pending | running | done | failed | cancelled
    settings_overrides: dict = field(default_factory=dict)
    exit_code: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
