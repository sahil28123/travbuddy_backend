"""
ARI — the calendar's reads and writes.

Availability, Rates and Inventory. Four operations, all of them thin: every one
is a single call into a Postgres function, and the reason is the same reason
`availability.py` reads `property_availability()` rather than joining the tables
itself.

**The inheritance is in SQL, once.** A night with no `room_inventory` row
inherits `room_types.total_units`; a night with no `rate_plan_rates` row
inherits `rate_plans.base_price_paise`. Materialising that here would be a
second implementation of the coalesce, and the two would disagree the first time
someone added a column.

**Writes go through `set_inventory` / `set_rates` and nowhere else.** Those
functions refuse to cut stock below what is already booked, apply a whole date
range in one statement, and treat null as "leave alone" so editing a minimum
stay does not freeze an inherited price into thirty rows. None of that survives
being reimplemented as a PATCH from a browser, which is why migration 7 dropped
the RLS write policy that made one possible.

What this module *does* own is the mapping from an HTTP error to a human
sentence. Postgres says `check_violation`; a revenue manager needs to be told
which night is already sold.
"""
import logging
import re
from datetime import date

from fastapi import HTTPException, status

from app.db.client import get_db, run_db

logger = logging.getLogger(__name__)

# A calendar window. Wider than a screen shows, because the grid pages forward
# and pre-fetches; narrower than the 400 nights the SQL allows, because nobody
# scrolls a year and a request that returns 400 × room types is a slow one.
MAX_WINDOW_DAYS = 120


def _window(from_date: date, days: int) -> tuple[str, str]:
    if days < 1 or days > MAX_WINDOW_DAYS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"A calendar window is between 1 and {MAX_WINDOW_DAYS} nights",
        )
    # property_availability and property_rates both take a half-open range and
    # generate_series to p_to - 1, so the exclusive end is `days` after the
    # start. Passing an inclusive end here would silently drop the last column
    # of every calendar.
    end = date.fromordinal(from_date.toordinal() + days)
    return from_date.isoformat(), end.isoformat()


async def get_inventory(property_id: str, from_date: date, days: int) -> list[dict]:
    start, end = _window(from_date, days)

    def _read():
        rows = get_db().rpc(
            "property_availability",
            {"p_property": property_id, "p_from": start, "p_to": end},
        ).execute().data or []
        return [
            {
                "roomTypeId": r["room_type_id"],
                "date": r["date"],
                "unitsTotal": r["units_total"],
                "unitsBooked": r["units_booked"],
                "unitsHeld": r["units_held"],
                "unitsBlocked": r["units_blocked"],
                "unitsLeft": r["units_left"],
                "stopSell": r["stop_sell"],
            }
            for r in rows
        ]

    return await run_db(_read)


async def get_rates(property_id: str, from_date: date, days: int) -> list[dict]:
    start, end = _window(from_date, days)

    def _read():
        rows = get_db().rpc(
            "property_rates",
            {"p_property": property_id, "p_from": start, "p_to": end, "p_audience": None},
        ).execute().data or []
        return [
            {
                "ratePlanId": r["rate_plan_id"],
                "date": r["date"],
                "pricePaise": r["price_paise"],
                "minStay": r["min_stay"],
                # Read with .get: older deployments of property_rates do not
                # return it, and the channel push simply omits maxNights then.
                "maxStay": r.get("max_stay"),
                "closedToArrival": r["closed_to_arrival"],
                "closedToDeparture": r["closed_to_departure"],
                "closed": r["closed"],
            }
            for r in rows
        ]

    return await run_db(_read)


def _readable(exc: Exception) -> HTTPException:
    """
    Turn a Postgres error into something a revenue manager can act on.

    The functions raise with deliberate errcodes and messages naming the night
    at fault. Losing that to a generic 500 would leave "why did my December
    price not save?" as the only available diagnosis.
    """
    text = str(exc)
    if "does not belong to property" in text:
        return HTTPException(status.HTTP_404_NOT_FOUND, "That room type or rate plan is not part of this property")
    if "already has" in text and "booked" in text:
        # The SQL names the night and the counts; pass it through rather than
        # replacing it with something vaguer.
        match = re.search(r"already has .*", text)
        detail = match.group(0).rstrip("\"'} \n") if match else text
        return HTTPException(status.HTTP_409_CONFLICT, f"Cannot reduce stock: this room type {detail}")
    for phrase in (
        "the end of the range is before its start",
        "may not span more than 400 nights",
        "a price cannot be negative",
        "a minimum stay is at least one night",
        "units_total cannot be negative",
        "closes the plan entirely",
    ):
        if phrase in text:
            return HTTPException(status.HTTP_400_BAD_REQUEST, phrase.capitalize())
    logger.exception("ARI write failed")
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not save the calendar change")


async def set_inventory(
    property_id: str,
    *,
    room_type_id: str,
    from_date: date,
    to_date: date,
    units_total: int | None,
    units_blocked: int | None,
    stop_sell: bool | None,
    oversell: int | None,
) -> int:
    """Returns the number of nights written."""

    def _write():
        try:
            return get_db().rpc(
                "set_inventory",
                {
                    "p_property": property_id,
                    "p_room_type": room_type_id,
                    "p_from": from_date.isoformat(),
                    "p_to": to_date.isoformat(),
                    "p_units_total": units_total,
                    "p_units_blocked": units_blocked,
                    "p_stop_sell": stop_sell,
                    "p_oversell": oversell,
                },
            ).execute().data or 0
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised as a readable one
            raise _readable(exc) from exc

    nights = await run_db(_write)
    # Out to the channels. Never raises — see services/repull/ari_push.py.
    from app.services.repull import ari_push
    await ari_push.enqueue_ari_push(property_id, room_type_id, from_date, to_date)
    return nights


async def set_rates(
    property_id: str,
    *,
    rate_plan_id: str,
    from_date: date,
    to_date: date,
    price_paise: int | None,
    min_stay: int | None,
    max_stay: int | None,
    closed_to_arrival: bool | None,
    closed_to_departure: bool | None,
    closed: bool | None,
) -> int:
    """Returns the number of nights written."""

    def _write():
        try:
            return get_db().rpc(
                "set_rates",
                {
                    "p_property": property_id,
                    "p_rate_plan": rate_plan_id,
                    "p_from": from_date.isoformat(),
                    "p_to": to_date.isoformat(),
                    "p_price": price_paise,
                    "p_min_stay": min_stay,
                    "p_max_stay": max_stay,
                    "p_cta": closed_to_arrival,
                    "p_ctd": closed_to_departure,
                    "p_closed": closed,
                },
            ).execute().data or 0
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised as a readable one
            raise _readable(exc) from exc

    nights = await run_db(_write)
    # Out to the channels, if this is the plan they sell. Never raises.
    from app.services.repull import ari_push
    await ari_push.enqueue_rate_plan(property_id, rate_plan_id, from_date, to_date)
    return nights
