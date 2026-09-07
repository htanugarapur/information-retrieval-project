"""Configuration loading.

Config is read once and treated as immutable. Every run artifact embeds the
resolved config, so a saved ranking can be reconstructed without guessing which
weights produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"


def project_root() -> Path:
    """Repository root — the directory containing config.yaml."""
    return Path(__file__).resolve().parent.parent


class Config(Mapping[str, Any]):
    """Read-only view over the parsed config tree.

    Mapping rather than a nested dataclass: the tree is deep and mostly passed
    straight through to components, and an immutable Mapping keeps the
    "never mutate config mid-run" rule enforceable rather than aspirational.
    """

    def __init__(self, data: Mapping[str, Any], source: Path | None = None):
        self._data = dict(data)
        self.source = source

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. 'retrieval.ppr.alpha'."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get_path(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted}")
        return value

    def resolve(self, dotted: str) -> Path:
        """Resolve a configured relative path against the project root."""
        raw = self.require(dotted)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else project_root() / candidate

    def to_dict(self) -> dict[str, Any]:
        """Deep copy, for embedding in run artifacts."""
        import copy

        return copy.deepcopy(self._data)


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml. Explicit path wins, then SYNAPSE_CONFIG, then default."""
    if path is None:
        path = os.environ.get("SYNAPSE_CONFIG") or project_root() / DEFAULT_CONFIG_NAME
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return Config(data, source=config_path)


@dataclass(frozen=True)
class Credentials:
    """API keys, read from the environment only — never from config.yaml."""

    s2_api_key: str | None
    openrouter_api_key: str | None

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key)


def load_credentials() -> Credentials:
    return Credentials(
        s2_api_key=os.environ.get("S2_API_KEY") or None,
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
    )
