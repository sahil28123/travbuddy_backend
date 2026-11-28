"""
Every Repull webhook event, once.

The same contract as `pms/webhook_log.py` for Razorpay, against its own table,
for the same reasons: a redelivery becomes a lookup instead of a re-execution
that has to be safe, and the table is the record of what Repull actually sent
when a hotel's calendar and an OTA's disagree.

Repull's dedupe key is `X-Repull-Event-Id`, stable across retries (the
per-attempt `X-Repull-Delivery-Id` is not). Repull retries a non-2xx up to five
times, a minute apart, so the release path below matters: an event whose
handler failed is marked `failed`, the route answers 500, and the retry can
claim it again.
"""
import logging

from app.db.client import get_db, run_db

logger = logging.getLogger(__name__)

TABLE = "repull_webhook_events"
RECLAIMABLE = "failed"


def _reclaim(event_id: str) -> bool:
    rows = (
        get_db().table(TABLE)
        .update({"status": "retrying"})
        .eq("event_id", event_id).eq("status", RECLAIMABLE)
        .execute().data or []
    )
    return bool(rows)


def _claim(event_id: str, event_type: str, workspace_id: str | None, payload: dict) -> bool:
    try:
        get_db().table(TABLE).insert({
            "event_id": event_id,
            "event_type": event_type,
            "workspace_id": workspace_id,
            "payload": payload,
        }).execute()
        return True
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "duplicate key" in message or "23505" in message:
            return _reclaim(event_id)
        # Logging must never stop an event being handled — every handler is
        # idempotent on its own key (the reservation's external id, the
        # message's external id), which is the real protection.
        logger.warning("could not log repull event %s: %s", event_id, type(exc).__name__)
        return True


async def claim(event_id: str | None, event_type: str, workspace_id: str | None,
                payload: dict) -> bool:
    """True if this event is new (or failed before) and should be handled."""
    if not event_id:
        return True
    return await run_db(_claim, event_id, event_type, workspace_id, payload or {})


def _mark(event_id: str, status: str, error: str | None) -> None:
    get_db().table(TABLE).update(
        {"status": status, "error": (error or "")[:400] or None}
    ).eq("event_id", event_id).execute()


async def mark(event_id: str | None, status: str, error: str | None = None) -> None:
    """`processed`, `ignored` or `failed`. Best-effort."""
    if not event_id:
        return
    try:
        await run_db(_mark, event_id, status, error)
    except Exception:  # noqa: BLE001
        logger.debug("could not mark repull event %s as %s", event_id, status)


async def release(event_id: str | None, reason: str) -> None:
    """Hand the event back so Repull's retry is handled rather than skipped."""
    await mark(event_id, RECLAIMABLE, reason)
