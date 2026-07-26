import os


async def run(account_id: str) -> dict:
    """Async is fine too - the dispatcher detects and awaits."""
    return {
        "agent": "billing",
        "account_id": account_id,
        "balance": 42.50,
        "region": os.environ.get("REGION", "unset"),
    }
