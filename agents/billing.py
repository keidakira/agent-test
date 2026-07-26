"""Billing agent: explains an account's standing by reading the ledger.

Async, because the dispatcher detects and awaits coroutines.
"""
import os

from . import _client

INSTRUCTIONS = """You are a billing agent.
Look up the account before saying anything about it — never guess a balance,
a plan, or an invoice. Report amounts in the account's own currency.
Flag `needs_review` when the balance is negative, an invoice is past due, or the
lookup fails, and explain why in the summary. Keep the summary to two sentences.
"""

# Stand-in for the billing datastore. Swap for the real client when it exists.
LEDGER = {
    "acct_1": {
        "balance": 42.50,
        "currency": "USD",
        "plan": "team",
        "invoices": [{"id": "inv_9", "amount": 42.50, "status": "open", "due": "2026-08-01"}],
    },
    "acct_2": {
        "balance": -18.00,
        "currency": "USD",
        "plan": "starter",
        "invoices": [{"id": "inv_7", "amount": 18.00, "status": "past_due", "due": "2026-07-01"}],
    },
}

TOOLS = [
    {
        "type": "function",
        "name": "lookup_account",
        "description": "Fetch balance, plan, and open invoices for an account id.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
    }
]

OUTPUT = _client.json_schema(
    "billing_result",
    {
        "summary": {"type": "string"},
        "balance": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "needs_review": {"type": "boolean"},
    },
    ["summary", "balance", "currency", "needs_review"],
)


def lookup_account(account_id: str) -> dict:
    record = LEDGER.get(account_id)
    if record is None:
        return {"found": False, "account_id": account_id}
    return {"found": True, "account_id": account_id, **record}


async def run(account_id: str) -> dict:
    client = _client.async_client()
    messages: list = [{"role": "user", "content": f"Summarize account {account_id}."}]

    for _ in range(4):
        response = await client.responses.create(
            model=_client.model(),
            instructions=INSTRUCTIONS,
            input=messages,
            tools=TOOLS,
            text=OUTPUT,
        )
        if not _client.has_tool_calls(response):
            break
        messages += _client.run_tool_calls(response, {"lookup_account": lookup_account})
    else:
        raise RuntimeError(f"billing agent did not settle on an answer for {account_id}")

    result = _client.parse_output(response)
    return {
        "agent": "billing",
        "account_id": account_id,
        "balance": result["balance"],
        "currency": result["currency"],
        "summary": result["summary"],
        "needs_review": result["needs_review"],
        "region": os.environ.get("REGION", "unset"),
    }
