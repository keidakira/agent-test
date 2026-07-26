"""Support agent: replies to a user message, keeping per-session context."""
from collections import OrderedDict

from . import _client

INSTRUCTIONS = """You are a customer support agent.
Reply directly to the user in a warm, plain tone — no greeting boilerplate, no
restating their message back to them. Two short paragraphs at most.
Set `escalate` when the request needs a human: refunds, account deletion, legal
or security matters, or anyone who is clearly upset. Classify the intent.
If you do not know something, say so and offer the next step.
"""

# session_id -> the last response id, so OpenAI holds the transcript for us.
# In-memory, so it resets on redeploy and is per-worker; move to Redis or the
# Conversations API if sessions must survive that.
MAX_SESSIONS = 1000
_SESSIONS: OrderedDict[str, str] = OrderedDict()

OUTPUT = _client.json_schema(
    "support_result",
    {
        "reply": {"type": "string"},
        "intent": {
            "type": "string",
            "enum": ["question", "bug", "billing", "feature_request", "complaint", "other"],
        },
        "escalate": {"type": "boolean"},
    },
    ["reply", "intent", "escalate"],
)


def _remember(session_id: str, response_id: str) -> None:
    _SESSIONS[session_id] = response_id
    _SESSIONS.move_to_end(session_id)
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def run(message: str, session_id: str = "anon") -> dict:
    previous = _SESSIONS.get(session_id)
    response = _client.client().responses.create(
        model=_client.model(),
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": message}],
        previous_response_id=previous,
        store=True,
        text=OUTPUT,
    )
    _remember(session_id, response.id)

    result = _client.parse_output(response)
    return {
        "agent": "support",
        "session_id": session_id,
        "reply": result["reply"],
        "intent": result["intent"],
        "escalate": result["escalate"],
    }
