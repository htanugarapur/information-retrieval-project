"""OpenRouter client for the two LLM explainers.

Every completion is cached in sqlite keyed by (model, prompt). That is what makes
an LLM-backed explainer reproducible: the first run costs a network call, every
replay is deterministic and offline. Re-running an experiment therefore cannot
silently change the numbers because a model drifted underneath it.

Free models are rate-limited and periodically retired, so the client falls down a
configured fallback chain and records WHICH model actually answered on every
response. A run artifact that does not name the responding model cannot be
audited, and mixed-model results reported as one number would be a lie.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import requests

from .db import Database, hash_prompt

log = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMUnavailable(RuntimeError):
    """No key, or every model in the chain refused. Never silently degraded."""


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool
    usage: dict[str, Any] | None = None
    attempts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "cached": self.cached,
            "usage": self.usage,
            "attempts": self.attempts,
        }


class OpenRouterClient:
    def __init__(
        self,
        db: Database,
        api_key: str | None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        fallback_models: Sequence[str] = (),
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_seconds: int = 120,
        max_retries: int = 4,
        seed: int = 0,
    ):
        self.db = db
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_models = list(fallback_models)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._session = requests.Session()
        self._random = random.Random(seed)
        self.calls = 0
        self.cache_hits = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def model_chain(self) -> list[str]:
        chain = [self.model] + [m for m in self.fallback_models if m != self.model]
        return chain

    # ------------------------------------------------------------ discovery

    def list_free_models(self) -> list[dict[str, Any]]:
        """Models currently priced at zero. Works without a key."""
        try:
            response = self._session.get(f"{self.base_url}/models", timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
        except (requests.RequestException, ValueError) as exc:
            log.warning("could not list models: %s", exc)
            return []

        free = []
        for entry in data:
            pricing = entry.get("pricing") or {}
            try:
                prompt_price = float(pricing.get("prompt", 1))
                completion_price = float(pricing.get("completion", 1))
            except (TypeError, ValueError):
                continue
            if prompt_price == 0 and completion_price == 0:
                free.append(
                    {
                        "id": entry.get("id"),
                        "name": entry.get("name"),
                        "context_length": entry.get("context_length"),
                    }
                )
        free.sort(key=lambda m: -(m.get("context_length") or 0))
        return free

    # ------------------------------------------------------------ completion

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        use_cache: bool = True,
        model: str | None = None,
    ) -> LLMResponse:
        """One completion, cache-first, falling down the model chain on failure."""
        chain = [model] if model else self.model_chain()
        cache_identity = chain[0]
        composite = f"{system or ''}\x1e{prompt}"

        if use_cache:
            # Cache is keyed on the FIRST model in the chain plus the prompt, so a
            # replay reuses the answer regardless of which fallback served it
            # originally. The stored record still names the true responder.
            cache_key = hash_prompt(cache_identity, composite)
            cached = self.db.get_llm_cache(cache_key)
            if cached:
                self.cache_hits += 1
                return LLMResponse(
                    text=cached["completion"],
                    model=cached["model"],
                    cached=True,
                    usage=cached["usage"],
                )

        if not self.api_key:
            raise LLMUnavailable(
                "OPENROUTER_API_KEY is not set. Create a key at "
                "https://openrouter.ai/keys and export it, or run with the "
                "metapath explainer only."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        attempts: list[str] = []
        last_error: str | None = None

        for candidate_model in chain:
            for attempt in range(self.max_retries):
                attempts.append(f"{candidate_model}#{attempt + 1}")
                try:
                    self.calls += 1
                    response = self._session.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/synapse-ir",
                            "X-Title": "Synapse IR",
                        },
                        json={
                            "model": candidate_model,
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        },
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:
                    last_error = str(exc)
                    self._sleep(attempt)
                    continue

                if response.status_code == 200:
                    try:
                        payload = response.json()
                        text = payload["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError) as exc:
                        last_error = f"malformed response: {exc}"
                        self._sleep(attempt)
                        continue

                    served_by = payload.get("model", candidate_model)
                    usage = payload.get("usage")
                    if use_cache:
                        self.db.put_llm_cache(
                            hash_prompt(cache_identity, composite),
                            served_by,
                            composite,
                            text,
                            usage,
                        )
                    return LLMResponse(
                        text=text,
                        model=served_by,
                        cached=False,
                        usage=usage,
                        attempts=attempts,
                    )

                if response.status_code in (429, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    self._sleep(attempt)
                    continue

                # 400/401/404 against this model: move to the next one rather
                # than burning retries on a model that will never answer.
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                break

        raise LLMUnavailable(
            f"all models failed ({', '.join(chain)}). last error: {last_error}"
        )

    def _sleep(self, attempt: int) -> None:
        delay = min(2.0 * (2 ** attempt), 30.0)
        delay += self._random.uniform(0, 1.0)
        time.sleep(delay)

    # -------------------------------------------------------- structured out

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        use_cache: bool = True,
    ) -> tuple[dict[str, Any] | None, LLMResponse]:
        """Completion parsed as JSON, with the raw response always returned.

        Returns (None, response) on unparseable output rather than raising: a
        model that fails to follow the schema is a RESULT about that explainer,
        and the harness records it as such instead of dropping the case.
        """
        response = self.complete(prompt, system=system, use_cache=use_cache)
        return parse_json_object(response.text), response


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a chat completion.

    Tolerates fenced blocks and leading prose, because free models fence
    inconsistently. Does NOT tolerate inventing values -- if nothing parses,
    the caller learns that and records it.
    """
    if not text:
        return None

    candidates: list[str] = []
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
