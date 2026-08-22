"""Marketing agent: conducts web research and generates high-engagement Twitter/X marketing content."""
from __future__ import annotations

from . import _client

INSTRUCTIONS = """You are an elite marketing strategist and copywriter specializing in high-growth Twitter/X content.

Your goal is to take a prompt or topic, use the web search tool to gather the freshest facts, industry trends, statistics, or recent developments, and craft high-performing marketing content for Twitter/X.

Guidelines:
- Always use the web_search tool to find accurate, timely data, real-world examples, or context before drafting content.
- Tone: Punchy, authoritative, insightful, and audience-centric (avoid corporate jargon and generic fluff).
- Single Tweet: Standalone, highly shareable post (< 280 characters) with a strong hook and clear value proposition.
- Thread: 3 to 7 sequential tweets designed for maximum engagement and retention:
  * Tweet 1: Scroll-stopping hook explaining what the reader will learn.
  * Middle Tweets: Core value, data points, breakdown of findings with citations/references from web research.
  * Final Tweet: Summary takeaway + strong Call to Action (CTA).
- Key Insights: 2-4 concise bullet points summarizing the underlying research.
- Hashtags: 3-5 relevant, high-traffic hashtags.
"""

OUTPUT = _client.json_schema(
    "marketing_content",
    {
        "hook": {
            "type": "string",
            "description": "Primary attention-grabbing headline or opening hook.",
        },
        "single_tweet": {
            "type": "string",
            "description": "A standalone, punchy tweet under 280 characters.",
        },
        "thread": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A structured multi-tweet thread (3 to 7 tweets).",
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key takeaways and research-backed data points discovered during search.",
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "url": {"type": ["string", "null"]},
                },
                "required": ["claim", "url"],
                "additionalProperties": False,
            },
            "description": "Web sources and claims cited in the content.",
        },
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recommended hashtags for the campaign.",
        },
        "call_to_action": {
            "type": "string",
            "description": "The concluding CTA to drive audience engagement or conversion.",
        },
    },
    [
        "hook",
        "single_tweet",
        "thread",
        "key_insights",
        "sources",
        "hashtags",
        "call_to_action",
    ],
)


def run(prompt: str) -> dict:
    """Single entry point for the marketing agent.

    Args:
        prompt: Topic, campaign objective, or product description to market.

    Returns:
        Structured dictionary containing research findings, tweet, thread, and metadata.
    """
    client = _client.client()
    response = client.responses.create(
        model=_client.model(),
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search"}],
        text=OUTPUT,
    )
    result = _client.parse_output(response)
    return {
        "agent": "marketing",
        "prompt": prompt,
        "model": response.model,
        "hook": result["hook"],
        "single_tweet": result["single_tweet"],
        "thread": result["thread"],
        "key_insights": result["key_insights"],
        "sources": result["sources"],
        "hashtags": result["hashtags"],
        "call_to_action": result["call_to_action"],
    }
