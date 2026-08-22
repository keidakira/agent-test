"""MarketingAgent class wrapper matching agent.yaml entrypoint configuration."""
from __future__ import annotations

from typing import Any
from agents.marketing import run as marketing_run


class MarketingAgent:
    """Marketing agent class that conducts web research and generates Twitter content."""

    def run(self, prompt: str, **kwargs: Any) -> dict:
        """Run the marketing agent with the given prompt.

        Args:
            prompt: Campaign topic, idea, or product to create Twitter content for.

        Returns:
            Structured marketing content dictionary.
        """
        return marketing_run(prompt=prompt)
