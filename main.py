#!/usr/bin/env python3
"""Main entry point for the Twitter Marketing Agent.

Usage:
    python main.py "Announcing our new open source developer tool"
"""
from __future__ import annotations

import json
import sys
from agents import run


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python main.py <prompt>")
        print("Example: python main.py 'Top open-source LLM releases this week'")
        return 1

    prompt = " ".join(sys.argv[1:])
    print(f"Running marketing agent with prompt: {prompt!r}...\n")
    result = run(prompt)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
