"""Data models for the UltraSinger GUI."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class LLMProvider:
    """An LLM API provider configuration.

    API keys are stored in the system keyring under
    ``llm_api_key_{id}`` — never in this dataclass or config.json.
    """

    name: str = ""  # display name, e.g. "Groq Free"
    api_base_url: str = ""  # e.g. "https://api.groq.com/openai/v1"
    default_model: str = ""  # e.g. "qwen/qwen3-32b"
    is_default: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        """Serialize for JSON config (no API key!)."""
        return {
            "id": self.id,
            "name": self.name,
            "api_base_url": self.api_base_url,
            "default_model": self.default_model,
            "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMProvider":
        """Deserialize from JSON config."""
        return cls(
            id=data.get("id", uuid.uuid4().hex[:8]),
            name=data.get("name", ""),
            api_base_url=data.get("api_base_url", ""),
            default_model=data.get("default_model", ""),
            is_default=data.get("is_default", False),
        )


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
