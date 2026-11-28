"""
A hotel arriving from another PMS: its room types and its future bookings,
brought across in one call.

    connect(purpose="import")  the hotelier connects Mews / Cloudbeds / Guesty…
          ↓
    run_import(property_id)
          ├─ every Repull listing → a HostOS room type (matched by name, or
          │  created with the listing's room count and a public rate plan)
          │  and a row in repull_room_map
          └─ every booking that has not checked out yet → a HostOS reservation,
             through the same upsert the webhooks use

**Matched before created.** A hotel that set up "Deluxe Double" in HostOS
during onboarding and then imports a Mews room type called "Deluxe double" has
one room type, not two. The match is the same normalised-name rule
`connections.automap` uses — exact after case and punctuation, or nothing.

**Created through onboarding, not beside it.** A room type and its price are
made with `onboarding.add_room_type_with_price`, the function the hotelier's
own setup screen calls, so an imported room type passes the same checks (a
name, 1–999 rooms, a price above ₹0) and gets the same codes.

**The price is one the hotel already charges, or none.** HostOS will not sell a
room at ₹0, and an imported room type needs a public rate plan before its
bookings can be recorded (`reservations.rate_plan_id` is NOT NULL). The plan's
price is the listing's own base price when Repull has it in rupees, otherwise
the median nightly rate of the room type's imported rupee bookings — what the
room has actually been selling for. With neither, the room type is created and
mapped without a plan, its bookings are counted as skipped, and the log says
why: inventing a price would put it on the hotel's website and, through Repull,
on every channel.

**Idempotent.** Listings already mapped are counted as matched, reservations
already imported are found by their Repull id, so running it again — after the
hotelier fixes a price, or a week later to catch new bookings — creates nothing
twice.
"""
import logging
import statistics
from datetime import date

from fastapi import HTTPException, status

from app.db.client import get_db, run_db
from app.services.pms import onboarding
from app.services.repull import client as repull_client
from app.services.repull import connections, reservations
from app.services.repull.client import RepullError, call

logger = logging.getLogger(__name__)

# 100 pages × 100 bookings per listing. A room type with ten thousand future
# stays is not a hotel this import was written for, and stopping is better
# than paging for ever.
_MAX_PAGES = 100


def _partner(property_id: str) -> str | None:
    rows = (get_db().table("properties").select("id, partner_id")
            .eq("id", property_id).limit(1).execute().data or [])
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Property not found")
    return rows[0].get("partner_id")


async def _hotel_providers(workspace_id: str) -> dict[str, str]:
    """Repull hotel id → the PMS it came from (`mews`, `cloudbeds`…)."""
    from repull.api.hotels import list_hotels

    try:
        page = await call(list_hotels.asyncio_detailed, workspace_id=workspace_id)
    except RepullError as exc:
        # The provider only labels the bookings; the import does not need it.
        logger.warning("could not list Repull hotels for workspace %s: %s", workspace_id, exc)
        return {}
    return {str(h["id"]): str(h["provider"]).lower()
            for h in page.get("data") or [] if h.get("id") and h.get("provider")}


def _provider_for(listing: dict, hotels: dict[str, str], fallback: str | None) -> str | None:
    hotel = listing.get("hotel") or {}
    return hotels.get(str(hotel.get("id"))) or fallback


def _listing_param(listing_id: str):
    # The SDK types `listing_id` as an int; Repull listing ids are numeric
    # strings. Anything else is passed through rather than guessed at.
    return int(listing_id) if str(listing_id).isdigit() else listing_id


async def _bookings(workspace_id: str, listing_id: str, today: date) -> list[dict]:
    """Every booking on one listing that has not checked out yet."""
    from repull.api.reservations import list_reservations

    out, cursor = [], None
    for _ in range(_MAX_PAGES):
        kwargs = {"listing_id": _listing_param(listing_id), "check_out_after": today,
                  "limit": 100}
        if cursor:
            kwargs["cursor"] = cursor
        page = await call(list_reservations.asyncio_detailed,
                          workspace_id=workspace_id, **kwargs)
        out.extend(page.get("data") or [])
        pagination = page.get("pagination") or {}
        cursor = pagination.get("nextCursor")
        if not pagination.get("hasMore") or not cursor:
            break
    return out


def _price_paise(listing: dict, bookings: list[dict]) -> int | None:
    """A nightly price this hotel already charges for this room, in paise."""
    pricing = listing.get("pricing") or {}
    base = pricing.get("basePrice")
    if base and str(pricing.get("currency") or "").upper() == "INR":
        return round(float(base) * 100)

    nightly = []
    for b in bookings:
        total, currency = reservations._money(b)
        if currency != "INR" or not total or str(b.get("status")).lower() == "cancelled":
            continue
        try:
            nights = (date.fromisoformat(str(b["checkOut"])[:10])
                      - date.fromisoformat(str(b["checkIn"])[:10])).days
        except (KeyError, ValueError):
            continue
        if nights > 0:
            nightly.append(total * 100 / nights)
    return round(statistics.median(nightly)) if nightly else None


async def _create_room_type(property_id: str, partner_id: str, name: str, units: int,
                            price_paise: int | None) -> tuple[dict, str | None]:
    """A room type and its public plan, through onboarding. Returns (room, plan_id)."""
    if price_paise is not None:
        try:
            onboarding.check_price_paise(price_paise)
        except HTTPException:
            price_paise = None
    room = await onboarding.add_room_type_with_price(
        property_id=property_id, partner_id=partner_id, name=name,
        total_units=min(max(int(units or 1), 1), 999), max_occupancy=2, code=None,
        price_paise=price_paise, meal_plan="EP",
    )
    if price_paise is None:
        return room, None
    _, plans = await run_db(connections._room_types_and_plans, property_id)
    return room, connections._public_plan(plans, room["id"])


async def run_import(property_id: str) -> dict:
    """
    Import room types and future bookings from the PMS this hotel connected.

    Returns `{roomTypesCreated, roomTypesMatched, reservationsImported,
    reservationsSkipped}`. A booking counts as imported when this run created,
    updated or cancelled it, and as skipped when it was already up to date or
    could not be recorded (logged with the reason).
    """
    if not repull_client.enabled():
        raise HTTPException(status.HTTP_409_CONFLICT, "Repull is not enabled on this deployment")
    conn = await connections.connection(property_id)
    if not conn:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Connect this hotel's PMS through Repull first")
    workspace_id = conn["workspace_id"]
    partner_id = await run_db(_partner, property_id)
    today = date.today()

    listings = await connections.listings(property_id, include="pricing")
    hotels = await _hotel_providers(workspace_id)
    fallback = next((str(p.get("provider")).lower()
                     for p in conn.get("connected_providers") or []
                     if str(p.get("provider")).lower() in connections.PMS_PROVIDERS), None)

    rooms, plans = await run_db(connections._room_types_and_plans, property_id)
    mapped = {m["repull_listing_id"]: m for m in await connections.room_map(property_id)}
    taken = {m["room_type_id"] for m in mapped.values()}
    by_name = {connections._norm(r.get("name")): r for r in rooms if r["id"] not in taken}
    names = {connections._norm(r.get("name")) for r in rooms}

    counts = {"roomTypesCreated": 0, "roomTypesMatched": 0,
              "reservationsImported": 0, "reservationsSkipped": 0}
    work: list[tuple[dict, list[dict], str | None]] = []

    for listing in listings:
        listing_id = str(listing.get("id") or "")
        if not listing_id:
            continue
        bookings = await _bookings(workspace_id, listing_id, today)
        provider = _provider_for(listing, hotels, fallback)
        work.append((listing, bookings, provider))

        if listing_id in mapped:
            counts["roomTypesMatched"] += 1
            continue

        name = (listing.get("name") or listing.get("publicName") or "").strip() \
            or f"Room type {listing_id}"
        key = connections._norm(name)
        room, plan_id = by_name.pop(key, None), None
        if room:
            plan_id = connections._public_plan(plans, room["id"])
            counts["roomTypesMatched"] += 1
        else:
            if key in names:
                # The name belongs to a room type already mapped to another
                # listing — two PMS room types with one name. Kept apart.
                name = f"{name} ({listing_id})"
            try:
                room, plan_id = await _create_room_type(
                    property_id, partner_id, name, listing.get("unitCount") or 1,
                    _price_paise(listing, bookings),
                )
            except HTTPException as exc:
                # Onboarding's own refusal (a name it will not take, say). The
                # listing stays unmapped and its bookings count as skipped.
                logger.warning("could not create a room type for listing %s: %s",
                               listing_id, exc.detail)
                continue
            names.add(connections._norm(name))
            counts["roomTypesCreated"] += 1
            if not plan_id:
                logger.warning(
                    "imported room type %s (listing %s) has no price in rupees; its "
                    "bookings are skipped until it has a rate plan", room["id"], listing_id,
                )
        saved = await run_db(connections._upsert_map, property_id, [{
            "room_type_id": room["id"],
            "rate_plan_id": plan_id,
            "repull_listing_id": listing_id,
            "listing_name": listing.get("name"),
            "unit_count": max(int(listing.get("unitCount") or 1), 1),
        }])
        mapped[listing_id] = saved[0]

    touched: dict[str, str] = {}
    for listing, bookings, provider in work:
        for booking in bookings:
            try:
                result = await reservations.upsert_external_reservation(
                    property_id, booking, source_override=provider, push=False,
                )
            except Exception:  # noqa: BLE001 — one bad booking must not stop the rest
                logger.warning("could not import Repull reservation %s", booking.get("id"),
                               exc_info=True)
                counts["reservationsSkipped"] += 1
                continue
            if result["outcome"] in ("created", "updated", "cancelled"):
                counts["reservationsImported"] += 1
                room_type_id = mapped[str(listing["id"])]["room_type_id"]
                check_out = str(booking.get("checkOut") or "")[:10]
                touched[room_type_id] = max(touched.get(room_type_id, ""), check_out)
            else:
                counts["reservationsSkipped"] += 1

    # One push per room type over everything the import changed, rather than
    # one per booking.
    for room_type_id, last in touched.items():
        await reservations._push(property_id, room_type_id, today,
                                 date.fromisoformat(last) if last else today)

    logger.info("Repull import for %s: %s", property_id, counts)
    return counts
