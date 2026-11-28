"""
The one place an API key becomes a request to Repull.

Every call goes through the official `repull-sdk` package — it builds the
request, owns the transport and keeps the paths right as the API grows. What
this module adds is the part an SDK cannot know about HostOS:

**Which workspace.** One HostOS deployment serves many hotels from one
`REPULL_API_KEY`. Each hotel is its own Repull sub-workspace, selected per
request with `X-Workspace-Id`, so Hotel A's Airbnb account is invisible to
Hotel B's calls by construction rather than by every caller remembering a
filter.

**What a failure means.** Repull answers every error with the same envelope —
`{"error": {"code", "message", "fix", "docs_url"}}` — and the `fix` field is
the sentence an operator actually needs. `RepullError` keeps it, and keeps
whether the call is worth retrying, so a Celery task can tell "try again in a
minute" (429, 5xx, a timeout) from "this will never work" (a revoked key, a
listing that is not ours) without parsing prose.

**Plain dicts out.** `call()` returns the decoded JSON body rather than the
SDK's attrs models. The models are generated from the OpenAPI spec and are
right today; a dict is still right on the day Repull adds a field this code has
never heard of, and every caller here reads three or four keys.
"""
import json
import logging
from typing import Any, Awaitable, Callable

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Timeouts are generous on read: an availability write is acknowledged only
# after Repull has pushed to every connected channel, and Booking.com alone can
# take several seconds.
_TIMEOUT = httpx.Timeout(45.0, connect=10.0)


class RepullError(Exception):
    """
    A Repull call that did not succeed, with the reason in a usable shape.

    `code` is Repull's machine code (`invalid_params`, `rate_limit_exceeded`,
    `managed_by_pms`, …) or one of ours for transport failures. `fix` is
    Repull's own sentence about what to do, passed through untouched — it is
    better advice than anything a generic mapping here could invent.
    """

    def __init__(self, code: str, message: str, *, status: int | None = None,
                 fix: str | None = None, docs_url: str | None = None,
                 retryable: bool = False):
        self.code = code
        self.message = message
        self.status = status
        self.fix = fix
        self.docs_url = docs_url
        self.retryable = retryable
        super().__init__(f"{code}: {message}" + (f" — {fix}" if fix else ""))


def enabled() -> bool:
    """Repull is on for this deployment: flag set and a key to call with."""
    return bool(settings.enable_hotel_os and settings.enable_repull and settings.repull_api_key)


def client(workspace_id: str | None = None):
    """
    An SDK client for the operator's key, scoped to one hotel's workspace.

    `workspace_id=None` is the operator's own workspace — used only to create
    a hotel's workspace in the first place and to register the webhook.
    """
    if not settings.repull_api_key:
        raise RepullError(
            "not_configured", "REPULL_API_KEY is not set",
            fix="Create a key at https://repull.dev/dashboard/keys and set REPULL_API_KEY.",
        )
    from repull import AuthenticatedClient

    headers = {"User-Agent": "HostOS (travbuddy_backend) repull-sdk"}
    if workspace_id:
        headers["X-Workspace-Id"] = workspace_id
    return AuthenticatedClient(
        base_url=settings.repull_base_url.rstrip("/"),
        token=settings.repull_api_key,
        headers=headers,
        timeout=_TIMEOUT,
    )


def _error_from(status: int, content: bytes) -> RepullError:
    try:
        body = json.loads(content or b"{}")
    except ValueError:
        body = {}
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        err = {}
    return RepullError(
        str(err.get("code") or f"http_{status}"),
        str(err.get("message") or (content or b"")[:200].decode(errors="ignore") or "Repull error"),
        status=status,
        fix=err.get("fix"),
        docs_url=err.get("docs_url"),
        # 409 idempotency_key_in_use is the first call still running: worth a retry.
        retryable=status == 429 or status >= 500 or err.get("code") == "idempotency_key_in_use",
    )


async def call(fn: Callable[..., Awaitable[Any]], *, workspace_id: str | None = None,
               **kwargs) -> Any:
    """
    Run one SDK `asyncio_detailed` function and return the JSON body.

        await call(update_availability.asyncio_detailed,
                   workspace_id=ws, property_id=123, body=req)

    Raises `RepullError` for any non-2xx answer and for transport failures
    (as `network_error`, retryable).
    """
    try:
        async with client(workspace_id) as c:
            response = await fn(client=c, **kwargs)
    except RepullError:
        raise
    except httpx.HTTPError as exc:
        raise RepullError("network_error", type(exc).__name__, retryable=True) from exc

    status = int(response.status_code)
    if status >= 400:
        error = _error_from(status, response.content)
        logger.warning("repull %s failed: %s", getattr(fn, "__module__", fn), error)
        raise error
    if not response.content:
        return {}
    try:
        return json.loads(response.content)
    except ValueError:
        return {}
