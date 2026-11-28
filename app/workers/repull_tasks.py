"""
The worker that tells the channels what the calendar says.

Two tasks, both on the `hotelos` queue through `travbuddy.pms.*`: they read
`room_inventory`, `rate_plan_rates` and `repull_room_map`, which exist only in
the hotel OS's database, so the marketplace worker must never pick them up.

`repull_push` is safe to run twice, late, or out of order — it carries a date
range, not values, and re-reads the calendar when it runs (see
`services/repull/ari_push.py`). That is what makes retrying it with backoff
correct rather than merely convenient: the tenth attempt sends today's numbers,
not the ones from the edit that queued the first.
"""
import asyncio
import logging

from app.services.repull.client import RepullError
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="travbuddy.pms.repull_push",
    bind=True,
    # Only a retryable RepullError ever leaves `push_room_type` — a 429, a 5xx
    # or a network failure (which `call()` reports as `network_error`). A
    # refusal about one group of nights is logged inside the push and does not
    # reach here, so autoretrying on the class retries exactly what is worth it.
    autoretry_for=(RepullError,),
    retry_backoff=10,
    retry_backoff_max=600,
    retry_jitter=True,
    # ~6 attempts over roughly ten minutes. Past that, Repull is having a bad
    # hour and the nightly resync is the better retry.
    max_retries=6,
    acks_late=True,
)
def repull_push_task(self, property_id: str, room_type_id: str,
                     from_date: str, to_date: str) -> dict:
    from app.services import trace
    from app.services.repull import ari_push

    with trace.span(request_id=f"repull-push-{self.request.id}"):
        return asyncio.run(ari_push.push_room_type(property_id, room_type_id,
                                                   from_date, to_date))


@celery_app.task(
    name="travbuddy.pms.repull_resync_all",
    bind=True,
    # No retry: it only enqueues, and runs again tomorrow.
    max_retries=0,
    acks_late=True,
)
def repull_resync_all_task(self) -> dict:
    """Queue a full calendar push for every hotel with an active Repull connection."""
    from app.services import trace
    from app.services.repull import ari_push

    with trace.span(request_id=f"repull-resync-{self.request.id}"):
        try:
            return asyncio.run(ari_push.resync_all())
        except Exception:  # noqa: BLE001 — a red task nobody watches helps nobody
            logger.exception("the nightly Repull resync failed")
            return {"properties": 0, "queued": 0, "error": "resync failed"}
