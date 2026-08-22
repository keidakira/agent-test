"""Shared OpenAI wiring for every agent in this package.

The local package is named `agents`, which shadows the OpenAI Agents SDK's
top-level `agents` module. We talk to the `openai` SDK directly instead.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = "gpt-4.1-mini"


def model() -> str:
    return os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or DEFAULT_MODEL


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env at the repo root "
            "(see .env.example) or export it in the environment."
        )
    return key


@lru_cache(maxsize=1)
def client() -> OpenAI:
    return OpenAI(api_key=_api_key())


@lru_cache(maxsize=1)
def async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_api_key())


def json_schema(name: str, properties: dict, required: list[str]) -> dict:
    """A `text.format` block for the Responses API, in strict mode."""
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
    }


def parse_output(response: Any) -> dict:
    """Structured output comes back as a JSON string on `output_text`."""
    text = (getattr(response, "output_text", None) or "").strip()
    if not text and hasattr(response, "output"):
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) == "message":
                for content_part in getattr(item, "content", []):
                    if getattr(content_part, "type", None) in ("text", "output_text"):
                        text = (getattr(content_part, "text", "") or "").strip()
                        if text:
                            break
    if not text:
        raise RuntimeError("Model returned an empty response.")

    # Strip markdown code blocks if wrapped
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        try:
            start_idx = text.find("{") if "{" in text else 0
            obj, _ = json.JSONDecoder().raw_decode(text[start_idx:])
            return obj
        except Exception as e:
            raise RuntimeError(f"Failed to parse model JSON output: {e}\nRaw output: {text}") from e



def run_tool_calls(response: Any, handlers: dict[str, Callable[..., Any]]) -> list[dict]:
    """Execute any function calls in `response` and build the follow-up input.

    Returns the items to append to the next `responses.create` input list:
    the model's own output items followed by one output item per call.
    """
    items: list[dict] = [item.model_dump() for item in response.output]
    for item in response.output:
        if item.type != "function_call":
            continue
        handler = handlers.get(item.name)
        result = (
            handler(**json.loads(item.arguments))
            if handler
            else {"error": f"unknown tool: {item.name}"}
        )
        items.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result),
            }
        )
    return items


def has_tool_calls(response: Any) -> bool:
    return any(item.type == "function_call" for item in response.output)
