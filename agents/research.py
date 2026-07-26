"""Research agent: answers a query, using web search when it needs fresh facts."""
from . import _client

INSTRUCTIONS = """You are a research agent.
Answer the user's query with specific, verifiable findings. Use the web search
tool whenever the query touches on current events, prices, versions, or anything
that may have changed recently. Each finding is one self-contained sentence.
Cite the source URL on a finding when it came from a search result.
If the evidence is thin, say so in the summary rather than padding the findings.
"""

OUTPUT = _client.json_schema(
    "research_result",
    {
        "summary": {"type": "string", "description": "Two or three sentence answer."},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source": {
                        "type": ["string", "null"],
                        "description": "URL, or null if not from a search result.",
                    },
                },
                "required": ["claim", "source"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    ["summary", "findings", "confidence"],
)


def run(query: str) -> dict:
    response = _client.client().responses.create(
        model=_client.model(),
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": query}],
        tools=[{"type": "web_search"}],
        text=OUTPUT,
    )
    result = _client.parse_output(response)
    return {
        "agent": "research",
        "query": query,
        "model": response.model,
        "summary": result["summary"],
        "findings": result["findings"],
        "confidence": result["confidence"],
    }
