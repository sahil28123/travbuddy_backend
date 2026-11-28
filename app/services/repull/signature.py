"""
What makes a Repull webhook body trustworthy.

Repull signs every delivery:

    X-Repull-Signature: t=1764288000,v1=5f2c…

    v1 = hex(HMAC_SHA256(secret, f"{t}.{raw_body}"))

where `secret` is the whole `whsec_…` string returned once when the webhook
was registered. See https://repull.dev/docs/webhooks/verify-signatures.

**The raw body, not the parsed one.** Re-serialising JSON changes key order and
whitespace, and a signature over different bytes is a different signature. The
route hands this module `await request.body()` before anything parses it.

**The timestamp is part of what is signed, and it is checked.** Without the
window a captured delivery — a reservation, say — could be replayed at this
endpoint forever. Five minutes either way absorbs clock skew and Repull's own
retry latency; a retry later than that arrives with a fresh `t` anyway, because
Repull signs each attempt.

**Constant-time comparison.** `hmac.compare_digest`, never `==`.
"""
import hashlib
import hmac
import time

TOLERANCE_SECONDS = 300


class SignatureError(Exception):
    """The delivery is not provably from Repull. The route answers 401."""


def _parse(header: str) -> tuple[int, list[str]]:
    timestamp: int | None = None
    signatures: list[str] = []
    for part in (header or "").split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            try:
                timestamp = int(value)
            except ValueError as exc:
                raise SignatureError("timestamp is not a number") from exc
        elif key == "v1" and value:
            signatures.append(value)
    if timestamp is None or not signatures:
        raise SignatureError("signature header is missing t= or v1=")
    return timestamp, signatures


def sign(secret: str, raw_body: bytes, timestamp: int | None = None) -> str:
    """The header Repull would send for this body. Used by tests and local replays."""
    t = int(time.time()) if timestamp is None else timestamp
    digest = hmac.new(secret.encode(), f"{t}.".encode() + raw_body, hashlib.sha256).hexdigest()
    return f"t={t},v1={digest}"


def verify(secret: str | None, raw_body: bytes, header: str | None,
           *, now: float | None = None) -> None:
    """Raise `SignatureError` unless `header` is a valid, fresh signature of `raw_body`."""
    if not secret:
        raise SignatureError("REPULL_WEBHOOK_SECRET is not configured")
    if not header:
        raise SignatureError("X-Repull-Signature header is missing")
    timestamp, signatures = _parse(header)
    current = time.time() if now is None else now
    if abs(current - timestamp) > TOLERANCE_SECONDS:
        raise SignatureError("signature timestamp is outside the 5-minute window")
    expected = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + raw_body, hashlib.sha256,
    ).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in signatures):
        raise SignatureError("signature does not match the body")
