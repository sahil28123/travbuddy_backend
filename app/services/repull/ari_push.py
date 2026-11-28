"""
The calendar, out: HostOS availability and rates pushed to every channel.

    set_inventory / set_rates / a booking / a cancel
        └─► enqueue_ari_push(property, room_type, from, to)     never raises
              └─► travbuddy.pms.repull_push   (hotelos queue, retried)
                    └─► push_room_type()  read effective nights → group → PUT

**HostOS is the source of truth, and the push is a snapshot of it.** The task
does not carry "price changed to ₹4,200" across the queue; it carries "these
nights of this room type changed" and reads the calendar again when it runs.
That is what makes it safe to run twice, run late or run out of order: whatever
runs last sends what the calendar says *then*, never what it said when someone
clicked save. It is also why the values come from `property_availability` and
`property_rates` — the same reads the calendar grid uses — so the inheritance
from `room_types.total_units` and `rate_plans.base_price_paise` is resolved in
SQL once, rather than reimplemented here and disagreeing later.

**A calendar edit never fails because Repull is down.** Enqueueing is wrapped
so that a missing broker, a bad key or an unmapped room logs a warning and
returns. The hotelier's save already happened in HostOS; the nightly resync and
the next edit will carry it out. A booking refused because a channel manager
was unreachable would be the integration costing the hotel the very sale it
exists to win.

**What a channel is told it can sell.** Free stock is

    units_total − units_booked − units_held − units_blocked

and deliberately not `units_left`, which also counts the oversell allowance.
Oversell is a front-desk judgement call ("we can always upgrade one to a
suite"); offering it to Booking.com turns that judgement into a commitment made
by a stranger at 3am. A night is `available` only with free stock, no stop-sell
and the mapped rate plan open.

`availableUnits` goes only to listings Repull knows sell several identical
rooms (`unit_count > 1`), and never more than that count — Repull refuses a
number above the listing's own with a 422. A single-unit listing (an Airbnb
home) is fully described by `available`, and sending it a unit count would be
noise at best. Where the room type is owned by a PMS on Repull's side, the
count is refused with `managed_by_pms`; the push then resends the same nights
without it, because open/closed, price and restrictions are still ours to set.

**One call per run of identical nights.** A year of a room type whose price
changes on weekends is a few dozen writes, not 365. Repull applies one set of
values to every date in a request, and caps a request at 731 dates (Airbnb's
own limit), so consecutive nights with the same values are grouped and long
runs are split.

**Failures are sorted, not flattened.** A `RepullError` marked `retryable`
(429, 5xx, a timeout) propagates so Celery retries the whole push with backoff
— re-reading the calendar each time, so a retry never sends stale numbers. Any
other refusal is about one group of nights and is logged with Repull's own
`fix` sentence, and the push carries on with the rest. A channel that rejected
part of an accepted write comes back in `synced.channels[]`; those are logged
and returned, because "Repull took it, Booking.com did not" is exactly the
line an operator needs to find.
"""
import asyncio
import logging
from datetime import date, timedelta

from app.config import settings
from app.db.client import get_db, run_db
from app.services.pms import ari as ari_reads
from app.services.repull import client as repull_client
from app.services.repull import connections
from app.services.repull.client import RepullError

logger = logging.getLogger(__name__)

# Repull (and Airbnb behind it) refuse a calendar write covering more dates.
MAX_DATES_PER_CALL = 731

PUSH_TASK = "travbuddy.pms.repull_push"


# ── enqueueing: cheap, quiet, never the reason a save fails ─────────────────

def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _mapped_room_type(room_type_id: str) -> bool:
    rows = (get_db().table("repull_room_map").select("room_type_id")
            .eq("room_type_id", room_type_id).limit(1).execute().data or [])
    return bool(rows)


def _room_type_for_plan(rate_plan_id: str) -> str | None:
    rows = (get_db().table("repull_room_map").select("room_type_id")
            .eq("rate_plan_id", rate_plan_id).limit(1).execute().data or [])
    return rows[0]["room_type_id"] if rows else None


def _send(property_id: str, room_type_id: str, start: str, end: str) -> None:
    from app.workers.repull_tasks import repull_push_task

    # `retry=False`: kombu's default is to keep retrying the publish for
    # seconds when Redis is unreachable, and those seconds would be spent
    # inside somebody's calendar save. Failing fast and logging is the point.
    repull_push_task.apply_async(args=(property_id, room_type_id, start, end), retry=False)


async def enqueue_ari_push(property_id: str, room_type_id: str, from_date, to_date) -> None:
    """
    Queue a push of `from_date`..`to_date` (inclusive) for one room type.

    A no-op unless Repull is enabled and the room type is mapped to a listing.
    Never raises: every failure here is logged and swallowed.
    """
    if not repull_client.enabled():
        return
    try:
        if not await run_db(_mapped_room_type, room_type_id):
            return
        start, end = _as_date(from_date), _as_date(to_date)
        if end < start:
            return
        await asyncio.to_thread(_send, property_id, room_type_id,
                                start.isoformat(), end.isoformat())
    except Exception:  # noqa: BLE001 — the caller's write already succeeded
        logger.warning(
            "could not queue a Repull push for room type %s (%s..%s); the nightly "
            "resync will carry it", room_type_id, from_date, to_date, exc_info=True,
        )


async def enqueue_rate_plan(property_id: str, rate_plan_id: str, from_date, to_date) -> None:
    """
    Queue a push after a rate edit — only if this plan is the one a channel sells.

    A room type can have several plans (B2B, member, refundable); only the one
    in `repull_room_map.rate_plan_id` reaches the OTAs, so editing an agent
    rate must not cost a year of calendar writes.
    """
    if not repull_client.enabled():
        return
    try:
        room_type_id = await run_db(_room_type_for_plan, rate_plan_id)
    except Exception:  # noqa: BLE001 — never into the caller
        logger.warning("could not resolve rate plan %s for a Repull push",
                       rate_plan_id, exc_info=True)
        return
    if room_type_id:
        await enqueue_ari_push(property_id, room_type_id, from_date, to_date)


async def enqueue_stay(property_id: str, room_type_id: str | None, check_in, check_out) -> None:
    """A stay moved stock: push its nights. `check_out` is exclusive, as on a stay."""
    if not room_type_id or not check_in or not check_out:
        return
    try:
        last_night = _as_date(check_out) - timedelta(days=1)
    except ValueError:
        return
    await enqueue_ari_push(property_id, room_type_id, check_in, last_night)


async def enqueue_room_type(property_id: str, room_type_id: str) -> None:
    """The next `repull_resync_days` of one room type — used right after mapping it."""
    today = date.today()
    await enqueue_ari_push(property_id, room_type_id, today,
                           today + timedelta(days=settings.repull_resync_days - 1))


async def enqueue_property(property_id: str, *, days: int) -> dict:
    """Queue every mapped room type of a hotel for the next `days` nights."""
    if not repull_client.enabled():
        return {"queued": 0, "roomTypes": [], "reason": "repull is not enabled"}
    today = date.today()
    end = today + timedelta(days=max(1, int(days)) - 1)
    rows = await connections.room_map(property_id)
    room_types = [r["room_type_id"] for r in rows]
    for room_type_id in room_types:
        await enqueue_ari_push(property_id, room_type_id, today, end)
    return {"queued": len(room_types), "roomTypes": room_types,
            "from": today.isoformat(), "to": end.isoformat()}


# ── the push ─────────────────────────────────────────────────────────────────

async def _read_nights(property_id: str, room_type_id: str, rate_plan_id: str | None,
                       start: date, end: date) -> tuple[dict, dict]:
    """
    Effective inventory and rates per night, paged through the calendar's own
    reads (which accept at most `MAX_WINDOW_DAYS` nights at a time).
    """
    inventory: dict[str, dict] = {}
    rates: dict[str, dict] = {}
    cursor = start
    while cursor <= end:
        days = min(ari_reads.MAX_WINDOW_DAYS, (end - cursor).days + 1)
        for row in await ari_reads.get_inventory(property_id, cursor, days):
            if str(row.get("roomTypeId")) == str(room_type_id):
                inventory[str(row["date"])[:10]] = row
        if rate_plan_id:
            for row in await ari_reads.get_rates(property_id, cursor, days):
                if str(row.get("ratePlanId")) == str(rate_plan_id):
                    rates[str(row["date"])[:10]] = row
        cursor += timedelta(days=days)
    return inventory, rates


def night_values(inv: dict, rate: dict | None, unit_count: int) -> dict:
    """What one night looks like to a channel, in Repull's field names."""
    total = int(inv.get("unitsTotal") or 0)
    taken = sum(int(inv.get(k) or 0) for k in ("unitsBooked", "unitsHeld", "unitsBlocked"))
    free = max(0, total - taken)
    rate = rate or {}
    values: dict = {
        "available": free > 0 and not inv.get("stopSell") and not rate.get("closed"),
    }
    if unit_count > 1:
        values["availableUnits"] = min(free, unit_count)
    if rate.get("pricePaise") is not None:
        # Paise in HostOS, rupees on the wire. Rounded to the paisa so a float
        # never turns ₹4,200 into ₹4,199.999999.
        values["price"] = round(int(rate["pricePaise"]) / 100, 2)
    if rate.get("minStay") is not None:
        values["minNights"] = int(rate["minStay"])
    if rate.get("maxStay") is not None:
        values["maxNights"] = int(rate["maxStay"])
    if rate.get("closedToArrival") is not None:
        values["closedToArrival"] = bool(rate["closedToArrival"])
    if rate.get("closedToDeparture") is not None:
        values["closedToDeparture"] = bool(rate["closedToDeparture"])
    return values


def group_nights(nights: list[tuple[date, dict]]) -> list[tuple[list[date], dict]]:
    """
    Runs of consecutive nights with identical values, each ≤ 731 dates.

    A gap (a night the read did not return) ends a run: grouping across it
    would write values to a date nobody looked at.
    """
    groups: list[tuple[list[date], dict]] = []
    for day, values in nights:
        if groups:
            dates, current = groups[-1]
            if (current == values and dates[-1] + timedelta(days=1) == day
                    and len(dates) < MAX_DATES_PER_CALL):
                dates.append(day)
                continue
        groups.append(([day], values))
    return groups


def _request(dates: list[date], values: dict):
    from repull.models.availability_write_request import AvailabilityWriteRequest

    names = {"available": "available", "price": "price", "minNights": "min_nights",
             "maxNights": "max_nights", "closedToArrival": "closed_to_arrival",
             "closedToDeparture": "closed_to_departure", "availableUnits": "available_units"}
    return AvailabilityWriteRequest(
        dates=list(dates), **{names[k]: v for k, v in values.items() if k in names},
    )


async def _write(workspace_id: str, listing_id: str, dates: list[date], values: dict) -> dict:
    from repull.api.availability import update_availability

    return await repull_client.call(
        update_availability.asyncio_detailed, workspace_id=workspace_id,
        property_id=int(listing_id), body=_request(dates, values),
    )


async def push_room_type(property_id: str, room_type_id: str, from_date, to_date) -> dict:
    """
    Send nights `from_date`..`to_date` (inclusive) of one room type to Repull.

    Returns a summary. Raises only a retryable `RepullError` — see the module
    docstring for why everything else is logged and the push continues.
    """
    summary: dict = {"roomTypeId": room_type_id, "groups": 0, "calls": 0,
                     "nights": 0, "failedChannels": [], "errors": []}
    if not repull_client.enabled():
        return {**summary, "skipped": "repull is not enabled"}
    mapping = await run_db(lambda: (get_db().table("repull_room_map").select("*")
                                    .eq("room_type_id", room_type_id).limit(1)
                                    .execute().data or [None])[0])
    if not mapping or not mapping.get("repull_listing_id"):
        return {**summary, "skipped": "room type is not mapped"}
    conn = await connections.connection(property_id)
    if not conn or not conn.get("workspace_id"):
        return {**summary, "skipped": "property is not connected"}

    listing_id = str(mapping["repull_listing_id"])
    unit_count = int(mapping.get("unit_count") or 1)
    start, end = _as_date(from_date), _as_date(to_date)
    summary["listingId"] = listing_id
    if end < start:
        return {**summary, "skipped": "empty range"}

    inventory, rates = await _read_nights(property_id, room_type_id,
                                          mapping.get("rate_plan_id"), start, end)
    nights = []
    day = start
    while day <= end:
        inv = inventory.get(day.isoformat())
        if inv is not None:
            nights.append((day, night_values(inv, rates.get(day.isoformat()), unit_count)))
        day += timedelta(days=1)

    groups = group_nights(nights)
    summary["groups"] = len(groups)
    summary["nights"] = len(nights)
    # Once Repull says the unit count belongs to a PMS, it says so for every
    # night of this listing — stop sending it rather than be refused per group.
    drop_units = False

    for dates, values in groups:
        if drop_units:
            values = {k: v for k, v in values.items() if k != "availableUnits"}
        try:
            try:
                result = await _write(conn["workspace_id"], listing_id, dates, values)
            except RepullError as exc:
                if exc.code != "managed_by_pms" or "availableUnits" not in values:
                    raise
                logger.info("listing %s: unit count is managed by a PMS; pushing "
                            "without availableUnits", listing_id)
                drop_units = True
                summary["calls"] += 1
                values = {k: v for k, v in values.items() if k != "availableUnits"}
                result = await _write(conn["workspace_id"], listing_id, dates, values)
            summary["calls"] += 1
        except RepullError as exc:
            if exc.retryable:
                raise
            summary["calls"] += 1
            summary["errors"].append({"from": dates[0].isoformat(), "to": dates[-1].isoformat(),
                                      "code": exc.code, "message": exc.message, "fix": exc.fix})
            logger.warning("Repull refused %s..%s for listing %s: %s",
                           dates[0], dates[-1], listing_id, exc)
            continue

        for channel in ((result or {}).get("synced") or {}).get("channels") or []:
            if channel.get("status") == "failed":
                failed = {"from": dates[0].isoformat(), "to": dates[-1].isoformat(),
                          "listingId": channel.get("listingId"),
                          "channel": channel.get("channel"), "error": channel.get("error")}
                summary["failedChannels"].append(failed)
                logger.warning("[repull-channel-failed] %s", failed)

    logger.info("pushed room type %s → listing %s: %d nights in %d calls, %d refused, "
                "%d channel failures", room_type_id, listing_id, summary["nights"],
                summary["calls"], len(summary["errors"]), len(summary["failedChannels"]))
    return summary


# ── nightly ──────────────────────────────────────────────────────────────────

def _active_properties() -> list[str]:
    rows = (get_db().table("repull_connections").select("property_id")
            .eq("status", "active").execute().data or [])
    return [r["property_id"] for r in rows]


async def resync_all() -> dict:
    """
    Queue a full push for every connected hotel.

    The far edge of the calendar moves forward a day every night, and a push
    lost to a broker outage leaves a channel wrong until somebody edits the
    same nights again. One nightly pass bounds both to a day.
    """
    if not repull_client.enabled():
        return {"properties": 0, "queued": 0}
    queued = 0
    properties = await run_db(_active_properties)
    for property_id in properties:
        try:
            queued += (await enqueue_property(property_id, days=settings.repull_resync_days))["queued"]
        except Exception:  # noqa: BLE001 — one hotel must not stop the rest
            logger.warning("could not queue the Repull resync for %s", property_id, exc_info=True)
    return {"properties": len(properties), "queued": queued}
