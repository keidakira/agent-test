"""A plain function. No platform imports, no decorators, no server."""
import os


def run(query: str) -> dict:
    return {
        "agent": "research",
        "query": query,
        "model": os.environ.get("MODEL", "unset"),
        "findings": [f"result for {query}"],
    }
