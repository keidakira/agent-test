#!/usr/bin/env python3
"""Call an agent from the shell, the way the platform would call it.

    python dev.py research query="best python http client"
    python dev.py billing account_id=acct_2
    python dev.py support message="my invoice looks wrong" session_id=s1
"""
import asyncio
import importlib
import inspect
import json
import sys
from pathlib import Path

import yaml


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    name, pairs = argv[0], argv[1:]
    spec = yaml.safe_load(Path(__file__).parent.joinpath("agents.yaml").read_text())
    agent = next((a for a in spec["agents"] if a["name"] == name), None)
    if agent is None:
        print(f"no agent named {name!r} in agents.yaml")
        return 2

    kwargs = dict(p.split("=", 1) for p in pairs)
    unknown = set(kwargs) - set(agent["input"])
    if unknown:
        print(f"{name} accepts {agent['input']}, got unexpected {sorted(unknown)}")
        return 2

    module, func = agent["entrypoint"].split(":")
    run = getattr(importlib.import_module(module), func)
    result = run(**kwargs)
    if inspect.iscoroutine(result):
        result = asyncio.run(result)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
