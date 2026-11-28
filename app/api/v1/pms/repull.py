"""
Selling a hotel on the channels: the Repull routes.

    POST /pms/properties/{id}/repull/connect   hosted Connect link (OTAs or a PMS)
    GET  /pms/properties/{id}/repull/status    what is connected, what is mapped
    GET  /pms/properties/{id}/repull/listings  the Repull listings to map onto
    PUT  /pms/properties/{id}/repull/map       room type ⇄ listing, by hand
    POST /pms/properties/{id}/repull/automap   room type ⇄ listing, by name
    POST /pms/properties/{id}/repull/resync    push the whole calendar again
    POST /pms/properties/{id}/repull/import    bring rooms + bookings from a PMS
    POST /pms/public/repull/webhook            everything Repull tells us

Mounted only when ENABLE_HOTEL_OS and ENABLE_REPULL are both set — see
`api/v1/router.py`. The guide is docs/repull.md.

**Connecting is the owner's decision** (`require_member("*")`): it hands an OTA
account's calendar to this system. Mapping and resyncing are `rates`, the same
permission that edits the calendar by hand, because that is what they do.

**The webhook is unauthenticated and signed.** The HMAC over the raw body is
the whole authentication story, exactly as for Razorpay. Every event is
claimed once in `repull_webhook_events`; a handler that fails releases it and
answers 500 so Repull's retry does the work again, and everything else answers
200 — retrying an event we have deliberately ignored is noise.
"""
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import settings
from app.core.limiter import limiter
from app.deps.property import require_member
from app.services.repull import connections, events
from app.services.repull.signature import SignatureError, verify

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pms", tags=["repull"])


# ── request bodies ───────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    purpose: Literal["connect", "import"] = "connect"
    redirect_url: str | None = Field(default=None, alias="redirectUrl")

    model_config = {"populate_by_name": True}


class RoomMapping(BaseModel):
    room_type_id: str = Field(alias="roomTypeId")
    listing_id: str | int | None = Field(default=None, alias="listingId")
    rate_plan_id: str | None = Field(default=None, alias="ratePlanId")

    model_config = {"populate_by_name": True}


class MapRequest(BaseModel):
    mappings: list[RoomMapping]


class ResyncRequest(BaseModel):
    days: int | None = Field(default=None, ge=1, le=731)


# ── setup ────────────────────────────────────────────────────────────────────

@router.post("/properties/{property_id}/repull/connect")
@limiter.limit("20/minute")
async def repull_connect(request: Request, property_id: str, body: ConnectRequest,
                         member=Depends(require_member("*"))):
    """
    A link to Repull's hosted Connect page for this hotel.

    `purpose: "connect"` offers Airbnb, Booking.com, Vrbo and tiket.com.
    `purpose: "import"` offers the PMSs a hotel may be moving from (Mews,
    Cloudbeds, Guesty, …); after it, call `/repull/import`.
    """
    return await connections.start_connect(
        property_id, purpose=body.purpose, redirect_url=body.redirect_url,
    )


@router.get("/properties/{property_id}/repull/status")
@limiter.limit("60/minute")
async def repull_status(request: Request, property_id: str,
                        member=Depends(require_member("rates"))):
    return await connections.refresh_status(property_id)


@router.get("/properties/{property_id}/repull/listings")
@limiter.limit("30/minute")
async def repull_listings(request: Request, property_id: str,
                          member=Depends(require_member("rates"))):
    rows = await connections.listings(property_id)
    return [{
        "id": r.get("id"), "name": r.get("name"), "kind": r.get("kind"),
        "unitCount": r.get("unitCount"),
        "channels": [c.get("platform") for c in (r.get("channels") or [])],
    } for r in rows]


@router.put("/properties/{property_id}/repull/map")
@limiter.limit("30/minute")
async def repull_map(request: Request, property_id: str, body: MapRequest,
                     member=Depends(require_member("rates"))):
    saved = await connections.save_map(property_id, [
        {"roomTypeId": m.room_type_id, "listingId": m.listing_id, "ratePlanId": m.rate_plan_id}
        for m in body.mappings
    ])
    from app.services.repull import ari_push
    for row in saved:
        await ari_push.enqueue_room_type(property_id, row["room_type_id"])
    return {"mapped": saved}


@router.post("/properties/{property_id}/repull/automap")
@limiter.limit("10/minute")
async def repull_automap(request: Request, property_id: str,
                         member=Depends(require_member("rates"))):
    result = await connections.automap(property_id)
    from app.services.repull import ari_push
    for row in result["mapped"]:
        await ari_push.enqueue_room_type(property_id, row["room_type_id"])
    return result


@router.post("/properties/{property_id}/repull/resync")
@limiter.limit("6/minute")
async def repull_resync(request: Request, property_id: str, body: ResyncRequest | None = None,
                        member=Depends(require_member("rates"))):
    from app.services.repull import ari_push
    days = (body.days if body else None) or settings.repull_resync_days
    return await ari_push.enqueue_property(property_id, days=days)


@router.post("/properties/{property_id}/repull/import")
@limiter.limit("4/minute")
async def repull_import(request: Request, property_id: str,
                        member=Depends(require_member("*"))):
    from app.services.repull import pms_import
    return await pms_import.run_import(property_id)


# ── the webhook ──────────────────────────────────────────────────────────────

RESERVATION_EVENTS = {"reservation.created", "reservation.updated", "reservation.cancelled"}
MESSAGE_EVENTS = {"reservation.message.received"}
CONNECTION_EVENTS = {"connect.session.completed", "account.created", "account.disconnected"}


async def _route(event_type: str, property_id: str, envelope: dict) -> str:
    """Hand one event to its handler. Returns how it went: processed | ignored."""
    if event_type in RESERVATION_EVENTS:
        from app.services.repull import reservations
        return await reservations.handle_event(property_id, envelope)
    if event_type in MESSAGE_EVENTS:
        from app.services.messaging.ota import webhook as ota_webhook
        return await ota_webhook.handle_event(property_id, envelope)
    if event_type in CONNECTION_EVENTS:
        await connections.refresh_status(property_id)
        return "processed"
    return "ignored"


@router.post("/public/repull/webhook")
async def public_repull_webhook(
    request: Request,
    x_repull_signature: str = Header(default=None),
    x_repull_event_id: str = Header(default=None),
):
    raw = await request.body()
    try:
        verify(settings.repull_webhook_secret, raw, x_repull_signature)
    except SignatureError as exc:
        logger.warning("[repull webhook] rejected: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad signature.")

    try:
        envelope = json.loads(raw or b"{}")
    except ValueError:
        return {"ok": True, "ignored": "unparseable"}

    event_type = envelope.get("event") or ""
    event_id = x_repull_event_id or envelope.get("eventId")
    workspace_id = envelope.get("workspaceId")

    if event_type == "repull.ping":
        return {"ok": True, "pong": True}

    property_id = await connections.property_for_workspace(workspace_id)
    if not property_id:
        # Not a hotel this deployment knows — another deployment on the same
        # key, or a workspace whose hotel was deleted. Nothing to retry.
        return {"ok": True, "ignored": "unknown workspace"}

    if not await events.claim(event_id, event_type, workspace_id, envelope):
        return {"ok": True, "handled": "already seen"}

    try:
        outcome = await _route(event_type, property_id, envelope)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[repull webhook] %s %s failed", event_type, event_id)
        await events.release(event_id, f"{type(exc).__name__}: {exc}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Handler failed; Repull will retry.")
    await events.mark(event_id, outcome)
    return {"ok": True, "handled": outcome}
