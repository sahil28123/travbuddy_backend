"""
Which Repull workspace is which hotel, and which Repull listing is which room type.

    property ──1:1── repull_connections.workspace_id
    room_type ──1:1── repull_room_map.repull_listing_id  (+ the rate plan sold there)

**A hotel is a workspace.** The first time a hotel connects, the Connect session
is created with `workspace: {name, externalRef: property_id}` and Repull makes
a sub-workspace under the operator's key. Every later call for that hotel sends
`X-Workspace-Id`. A webhook carries `workspaceId`, which is how an incoming
booking finds its hotel — see `property_for_workspace`.

**Listings come from the channels, not from here.** When a hotelier connects
Booking.com or tiket.com, Repull imports each room type as a listing with its
real `unitCount`; Airbnb and Vrbo listings arrive the same way. HostOS then
maps its own room types onto those listings — automatically by name where the
names agree, by hand where they do not. Creating listings from HostOS instead
would produce single-unit rentals, which is the wrong shape for a hotel selling
eight Deluxe rooms.

**One rate plan per room type goes to the channels.** A HostOS room type can
have several rate plans (B2B, member, refundable…); an OTA listing has one
nightly price. The map stores which plan is the public one, defaulting to the
plan whose audience is public, then the first.
"""
import logging
import re

from fastapi import HTTPException, status

from app.config import settings
from app.db.client import get_db, run_db
from app.services.repull.client import RepullError, call

logger = logging.getLogger(__name__)

CHANNEL_PROVIDERS = ["airbnb", "booking", "vrbo-login", "tiket"]
PMS_PROVIDERS = [
    "mews", "cloudbeds", "guesty", "hostaway", "hospitable", "ownerrez", "lodgify",
    "smoobu", "beds24", "igms", "bookingsync", "track", "rentalready", "ciirus",
]


# ── reads ────────────────────────────────────────────────────────────────────

def _connection(property_id: str) -> dict | None:
    rows = (get_db().table("repull_connections").select("*")
            .eq("property_id", property_id).limit(1).execute().data or [])
    return rows[0] if rows else None


async def connection(property_id: str) -> dict | None:
    return await run_db(_connection, property_id)


def _property_for_workspace(workspace_id: str) -> str | None:
    rows = (get_db().table("repull_connections").select("property_id")
            .eq("workspace_id", workspace_id).limit(1).execute().data or [])
    return rows[0]["property_id"] if rows else None


async def property_for_workspace(workspace_id: str | None) -> str | None:
    """The hotel a webhook is about, or None if this deployment does not know it."""
    if not workspace_id:
        return None
    return await run_db(_property_for_workspace, str(workspace_id))


def _room_map(property_id: str) -> list[dict]:
    return (get_db().table("repull_room_map").select("*")
            .eq("property_id", property_id).execute().data or [])


async def room_map(property_id: str) -> list[dict]:
    return await run_db(_room_map, property_id)


def _map_for_listing(listing_id: str) -> dict | None:
    rows = (get_db().table("repull_room_map").select("*")
            .eq("repull_listing_id", str(listing_id)).limit(1).execute().data or [])
    return rows[0] if rows else None


async def map_for_listing(listing_id) -> dict | None:
    """The room type a Repull listing is mapped to, if any."""
    if listing_id is None:
        return None
    return await run_db(_map_for_listing, str(listing_id))


# ── connecting a hotel ───────────────────────────────────────────────────────

def _property(property_id: str) -> dict:
    rows = (get_db().table("properties").select("id, partner_id, name")
            .eq("id", property_id).limit(1).execute().data or [])
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Property not found")
    return rows[0]


def _save_connection(property_id: str, partner_id: str | None, workspace_id: str) -> dict:
    row = {"property_id": property_id, "partner_id": partner_id,
           "workspace_id": str(workspace_id), "status": "pending"}
    return (get_db().table("repull_connections")
            .upsert(row, on_conflict="property_id").execute().data or [row])[0]


def _guest_agent_answers_ota(property_id: str) -> None:
    """
    Let the hotel's guest agent answer OTA threads.

    `agents.channels` is the hotel's own setting, and a guest agent provisioned
    before this integration existed was stored without `ota` — so its OTA
    messages would route to a person. Added once, when the hotel first connects,
    rather than by a blanket migration: only hotels that chose to sell on the
    OTAs change, and a hotel that later removes `ota` by hand keeps its choice,
    because this never runs again for it.
    """
    db = get_db()
    rows = (db.table("agents").select("id, channels")
            .eq("property_id", property_id).eq("agent_type", "guest")
            .execute().data or [])
    for agent in rows:
        channels = list(agent.get("channels") or [])
        if "ota" not in channels:
            db.table("agents").update({"channels": channels + ["ota"]}) \
                .eq("id", agent["id"]).execute()


async def start_connect(property_id: str, *, purpose: str = "connect",
                        redirect_url: str | None = None) -> dict:
    """
    A hosted Connect link for this hotel.

    `purpose="connect"` offers the OTAs; `purpose="import"` offers the PMSs a
    hotel might be moving from. The hotelier finishes the flow on
    connect.repull.dev and comes back to `redirect_url`.
    """
    from repull.api.connect import create_connect_session
    from repull.models.create_connect_session_body import CreateConnectSessionBody
    from repull.models.create_connect_session_body_workspace import (
        CreateConnectSessionBodyWorkspace,
    )

    target = redirect_url or settings.repull_connect_redirect_url
    if not target:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Set REPULL_CONNECT_REDIRECT_URL, or pass redirect_url, so Repull knows "
            "where to send the hotelier back to",
        )
    prop = await run_db(_property, property_id)
    existing = await connection(property_id)
    providers = PMS_PROVIDERS if purpose == "import" else CHANNEL_PROVIDERS

    kwargs = {"redirect_url": target, "state": property_id, "allowed_providers": providers}
    if not existing:
        # First connect: Repull creates the hotel's workspace in the same call.
        kwargs["workspace"] = CreateConnectSessionBodyWorkspace(
            name=prop.get("name") or f"HostOS property {property_id}",
            external_ref=property_id,
        )
    try:
        session = await call(
            create_connect_session.asyncio_detailed,
            workspace_id=existing["workspace_id"] if existing else None,
            body=CreateConnectSessionBody(**kwargs),
        )
    except RepullError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            exc.fix or exc.message) from exc

    workspace_id = (existing or {}).get("workspace_id") or session.get("workspaceId")
    if not workspace_id:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            "Repull did not return a workspace for this hotel")
    if not existing:
        await run_db(_save_connection, property_id, prop.get("partner_id"), workspace_id)
        try:
            await run_db(_guest_agent_answers_ota, property_id)
        except Exception:  # noqa: BLE001 — a person still sees the thread
            logger.warning("could not add the ota channel to %s's guest agent", property_id)
    return {"url": session.get("url"), "expiresAt": session.get("expiresAt"),
            "workspaceId": workspace_id, "providers": providers}


def _set_status(property_id: str, fields: dict) -> None:
    get_db().table("repull_connections").update(fields).eq("property_id", property_id).execute()


async def refresh_status(property_id: str) -> dict:
    """Ask Repull which channels this hotel has connected, and remember it."""
    from repull.api.connect import list_connections

    conn = await connection(property_id)
    if not conn:
        return {"connected": False, "connections": [], "rooms": []}
    data = await call(list_connections.asyncio_detailed, workspace_id=conn["workspace_id"])
    rows = data.get("data") or []
    providers = [{"provider": r.get("provider"), "status": r.get("status"),
                  "fixUrl": r.get("fixUrl")} for r in rows]
    active = any(r.get("status") == "active" for r in rows)
    await run_db(_set_status, property_id,
                 {"status": "active" if active else "pending", "connected_providers": providers})
    return {"connected": active, "workspaceId": conn["workspace_id"],
            "connections": providers, "rooms": await room_map(property_id)}


# ── room types ⇄ listings ────────────────────────────────────────────────────

async def listings(property_id: str, *, include: str | None = None) -> list[dict]:
    """
    Every Repull listing in this hotel's workspace (room types, homes).

    `include` is Repull's expansion list (`"pricing"`, `"details"`…), passed
    through for the callers that need more than the summary row.
    """
    from repull.api.listings import list_listings

    conn = await connection(property_id)
    if not conn:
        raise HTTPException(status.HTTP_409_CONFLICT, "Connect this hotel to Repull first")
    out, cursor = [], None
    for _ in range(50):  # 50 pages × 100 is far beyond any one hotel
        kwargs = {"limit": 100}
        if include:
            kwargs["include"] = include
        if cursor:
            kwargs["cursor"] = cursor
        page = await call(list_listings.asyncio_detailed,
                          workspace_id=conn["workspace_id"], **kwargs)
        out.extend(page.get("data") or [])
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            break
    return out


def _norm(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _room_types_and_plans(property_id: str) -> tuple[list[dict], list[dict]]:
    db = get_db()
    rooms = (db.table("room_types").select("id, name, total_units")
             .eq("property_id", property_id).execute().data or [])
    plans = (db.table("rate_plans").select("id, room_type_id, name, audience")
             .eq("property_id", property_id).execute().data or [])
    return rooms, plans


def _public_plan(plans: list[dict], room_type_id: str) -> str | None:
    mine = [p for p in plans if p.get("room_type_id") == room_type_id]
    for p in mine:
        if (p.get("audience") or "public") == "public":
            return p["id"]
    return mine[0]["id"] if mine else None


def _upsert_map(property_id: str, rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    for r in rows:
        r["property_id"] = property_id
    return (get_db().table("repull_room_map")
            .upsert(rows, on_conflict="room_type_id").execute().data or rows)


async def save_map(property_id: str, mappings: list[dict]) -> list[dict]:
    """
    Map room types to listings: `[{"roomTypeId", "listingId", "ratePlanId"?}]`.

    `listingId: null` unmaps. The listing must exist in this hotel's workspace
    — checked against Repull, so a typo is a 422 now rather than a calendar
    that silently never syncs.
    """
    available = {str(l.get("id")): l for l in await listings(property_id)}
    rooms, plans = await run_db(_room_types_and_plans, property_id)
    room_ids = {r["id"] for r in rooms}
    rows, unmap = [], []
    for m in mappings:
        room_type_id, listing_id = m.get("roomTypeId"), m.get("listingId")
        if room_type_id not in room_ids:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"Room type {room_type_id} is not part of this property")
        if listing_id is None:
            unmap.append(room_type_id)
            continue
        listing = available.get(str(listing_id))
        if not listing:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"Listing {listing_id} is not in this hotel's Repull workspace")
        rows.append({
            "room_type_id": room_type_id,
            "rate_plan_id": m.get("ratePlanId") or _public_plan(plans, room_type_id),
            "repull_listing_id": str(listing_id),
            "listing_name": listing.get("name"),
            "unit_count": int(listing.get("unitCount") or 1),
        })
    if unmap:
        await run_db(lambda: get_db().table("repull_room_map").delete()
                     .eq("property_id", property_id).in_("room_type_id", unmap).execute())
    return await run_db(_upsert_map, property_id, rows)


async def automap(property_id: str) -> dict:
    """
    Map every unmapped room type whose name matches exactly one listing.

    Deliberately conservative: an exact match after normalising case and
    punctuation, or nothing. A wrong automatic map sells the wrong room on
    Booking.com; a missing one is a line in the status response.
    """
    existing = {m["room_type_id"] for m in await room_map(property_id)}
    rooms, _ = await run_db(_room_types_and_plans, property_id)
    by_name: dict[str, list[dict]] = {}
    for listing in await listings(property_id):
        by_name.setdefault(_norm(listing.get("name")), []).append(listing)
    proposals, unmatched = [], []
    for room in rooms:
        if room["id"] in existing:
            continue
        candidates = by_name.get(_norm(room.get("name")), [])
        if len(candidates) == 1:
            proposals.append({"roomTypeId": room["id"], "listingId": candidates[0]["id"]})
        else:
            unmatched.append({"roomTypeId": room["id"], "name": room.get("name"),
                              "candidates": len(candidates)})
    mapped = await save_map(property_id, proposals) if proposals else []
    return {"mapped": mapped, "unmatched": unmatched}
