def run(message: str, session_id: str = "anon") -> dict:
    return {
        "agent": "support",
        "session_id": session_id,
        "reply": f"you said: {message}",
    }
