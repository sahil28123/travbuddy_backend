"""
Bookings, in: a reservation made on Airbnb, Booking.com, Vrbo or tiket.com
becomes a HostOS reservation, and keeps tracking the channel's copy.

    reservation.created ─┐
    reservation.updated ─┼─► upsert_external_reservation ─► reservations row
    reservation.cancelled┘           │                       + stock moved
                                     └─► ari_push            (other channels
                                                              see one room less)

**The channel already sold it, at the channel's price.** A booking from an OTA
is not a quote waiting for a guest's yes — the guest has paid Airbnb or holds a
Booking.com confirmation. So none of `create_quote`'s machinery applies: no
hold, no negotiation clamp, no re-pricing. What it does share with HostOS's own
bookings is the part that must never have two spellings: the stock counter and
the reservation row (`booking.book_stock`, `booking.insert_reservation`), and
the guest record (`booking._resolve_guest`).

**Keyed on the Repull reservation id.** `external_id` is Repull's id, which is
unique across Repull, so a replayed webhook, a `reservation.updated` that
arrives before its `reservation.created`, and a PMS import re-run all find the
same row. The lookup is on `(property_id, external_id)` rather than on
`external_source` as well, deliberately: a booking imported from Mews and later
updated by a webhook that names its channel would otherwise become two
reservations for one guest. The `(external_source, external_id)` unique index
is what turns two near-simultaneous first deliveries into one insert and one
retried 500.

**Pending means no stock, exactly as everywhere else in HostOS.**
`cancellation.HOLDS_INVENTORY` is `{"confirmed"}`: a pending reservation never
called `book_inventory`. An Airbnb request-to-book arrives as `pending`, is
stored as `pending`, and takes its room only when the channel confirms it.

**An overbooking is recorded, never dropped.** When the channel confirms a
booking HostOS has no room left for, the reservation is still inserted as
`confirmed` — the guest holds a confirmed booking on the channel and is going to
turn up; a HostOS that refused it would only hide that from the desk. The
reservation carries a note, an escalation reaches the hotel's team (kind
`other`), and the room-nights read as oversold in `reconcile_inventory`, which
is the honest picture. Because `book_inventory` refused, the counter was not
moved, and the payload records that (`hostos.stockBooked = false`) so nothing
later releases stock this booking never took — see `_cancel`.

**Money.** Repull reports the channel's total in major units with a currency.
INR becomes paise on the reservation; any other currency is kept untouched in
`external_payload` with a note, and the paise columns stay at zero rather than
carry a converted figure nobody agreed to.
"""
import logging
from datetime import date

from app.db.client import get_db, run_db
from app.services.pms import booking, cancellation, escalation
from app.services.repull import client as repull_client
from app.services.repull import connections
from app.services.repull.client import call

logger = logging.getLogger(__name__)

# Repull statuses that mean the guest has a room. `completed` is a stay that
# already happened (an import, or a late webhook) and occupied its nights.
_TAKES_STOCK = frozenset({"confirmed", "completed"})

# HostOS statuses this module may still change. A guest who has checked in is
# the desk's to manage; a channel edit arriving afterwards updates the payload
# and nothing else.
_EDITABLE = frozenset({"pending", "confirmed"})

# The keys of a Repull reservation worth keeping on the row. The full object
# can carry the whole conversation and financial breakdown; this is what a
# person debugging a booking reaches for.
_PAYLOAD_KEYS = (
    "id", "listingId", "platform", "status", "statusDetail", "confirmationCode",
    "checkIn", "checkOut", "unit", "occupancy", "totalPrice", "currency",
    "conversationId", "createdAt", "updatedAt",
)

_PLATFORM_NAMES = {
    "airbnb": "Airbnb", "booking": "Booking.com", "vrbo": "Vrbo", "tiket": "tiket.com",
    "direct": "Direct", "website": "Website", "owner": "Owner",
}


# ── reading a Repull reservation ─────────────────────────────────────────────

def _platform(obj: dict) -> str:
    return str(obj.get("platform") or "other").lower()


def _channel_name(source: str) -> str:
    return _PLATFORM_NAMES.get(source, source.title())


def _money(obj: dict) -> tuple[float | None, str | None]:
    financials = obj.get("financials") or {}
    total = obj.get("totalPrice")
    if total is None:
        total = financials.get("totalPrice")
    currency = obj.get("currency") or financials.get("currency")
    try:
        total = float(total) if total is not None else None
    except (TypeError, ValueError):
        total = None
    return total, (str(currency).upper() if currency else None)


def _guest(obj: dict) -> tuple[str, str | None, str | None]:
    guest = obj.get("primaryGuest") or {}
    name = " ".join(p for p in (guest.get("firstName"), guest.get("lastName")) if p).strip()
    return (name or "Guest"), (guest.get("phone") or None), (guest.get("email") or None)


def _trimmed(obj: dict) -> dict:
    out = {k: obj[k] for k in _PAYLOAD_KEYS if k in obj}
    financials = obj.get("financials") or {}
    if financials:
        out["financials"] = {k: financials.get(k)
                             for k in ("totalPrice", "currency", "paymentStatus")
                             if k in financials}
    return out


def _nights(check_in: date, check_out: date) -> list[str]:
    return [date.fromordinal(d).isoformat()
            for d in range(check_in.toordinal(), check_out.toordinal())]


def _night_rows(reservation_id: str, check_in: date, check_out: date,
                total_paise: int) -> list[dict]:
    """
    The per-night trail, from the channel's total.

    An even split with the remainder on the last night, so the nights sum to
    the stay — the same allocation `analytics._night_money` falls back to. Tax
    is left at zero because the channel reports one all-in figure and does not
    say how much of it is tax; inventing a split would put a number on the GST
    report nobody computed. A stay with no rupee total writes no nights: rows of
    zeros say nothing that the reservation does not already say.
    """
    days = _nights(check_in, check_out)
    if not days or total_paise <= 0:
        return []
    per = total_paise // len(days)
    rows = []
    for i, day in enumerate(days):
        amount = per if i < len(days) - 1 else total_paise - per * (len(days) - 1)
        rows.append({"reservation_id": reservation_id, "date": day,
                     "rate_paise": amount, "room_paise": amount, "tax_paise": 0})
    return rows


# ── the database side ────────────────────────────────────────────────────────

def _existing(property_id: str, external_id: str) -> dict | None:
    rows = (get_db().table("reservations").select("*")
            .eq("property_id", property_id).eq("external_id", external_id)
            .limit(1).execute().data or [])
    return rows[0] if rows else None


def _context(property_id: str, room_type_id: str) -> dict:
    """The property (partner, name) and the room type's name and plans."""
    db = get_db()
    prop = (db.table("properties").select("id, partner_id, name")
            .eq("id", property_id).limit(1).execute().data or [{}])[0]
    room = (db.table("room_types").select("id, name")
            .eq("id", room_type_id).limit(1).execute().data or [{}])[0]
    plans = (db.table("rate_plans").select("id, room_type_id, audience")
             .eq("room_type_id", room_type_id).execute().data or [])
    return {"partner_id": prop.get("partner_id"), "hotel": prop.get("name") or "",
            "room_name": room.get("name"), "plans": plans}


def _stock_booked(row: dict) -> bool:
    """
    Whether this reservation is counted in `room_inventory`.

    Read from the payload this module wrote. A row without the marker was
    written by something else, and then HostOS's own rule applies: confirmed
    holds stock, pending does not.
    """
    marker = ((row.get("external_payload") or {}).get("hostos") or {}).get("stockBooked")
    if marker is None:
        return row.get("status") in cancellation.HOLDS_INVENTORY
    return bool(marker)


def _notes(source: str, total: float | None, currency: str | None,
           overbooked: bool, room_name: str | None) -> str | None:
    lines = []
    if overbooked:
        lines.append(
            f"Overbooked: {_channel_name(source)} confirmed this booking but HostOS had no "
            f"{room_name or 'room'} left for these nights. Move a guest or contact the channel."
        )
    if total is not None and currency and currency != "INR":
        lines.append(
            f"Booked on {_channel_name(source)} for {currency} {total:,.2f}. "
            "Amounts are not converted to rupees."
        )
    return "\n".join(lines) or None


def _hostos(row: dict) -> dict:
    return dict((row.get("external_payload") or {}).get("hostos") or {})


def _write_new(property_id: str, row: dict, take_stock: bool, total_paise: int,
               note_for) -> dict:
    """
    Take the stock (when the booking holds a room) and insert the row.

    A refused `book_inventory` does not stop the insert: see the module
    docstring on overbookings. `note_for(booked)` is the reservation's note
    once it is known whether the room was there.
    """
    db = get_db()
    room, ci, co = row["room_type_id"], row["check_in"], row["check_out"]
    booked = take_stock and booking.book_stock(db, room, ci, co, 1)
    note = note_for(booked)

    row = {**row, "notes": note,
           "external_payload": {**row["external_payload"],
                                "hostos": {"stockBooked": booked, "note": note}}}
    row["guest_id"] = booking._resolve_guest(
        db, property_id, row["partner_id"], row["guest_name"],
        row.get("guest_phone"), row.get("guest_email"),
    )
    reservation = booking.insert_reservation(
        db, row,
        compensate=lambda: booking.release_stock(db, room, ci, co, 1) if booked else None,
    )
    _write_nights(db, reservation["id"], ci, co, total_paise)
    return reservation


def _write_nights(db, reservation_id: str, check_in: str, check_out: str,
                  total_paise: int, *, replace: bool = False) -> None:
    if replace:
        db.table("reservation_nights").delete().eq("reservation_id", reservation_id).execute()
    nights = _night_rows(reservation_id, date.fromisoformat(str(check_in)[:10]),
                         date.fromisoformat(str(check_out)[:10]), total_paise)
    if nights:
        db.table("reservation_nights").insert(nights).execute()


def _update(existing: dict, patch: dict, take_stock: bool, total_paise: int,
            note_for) -> tuple[dict, bool]:
    """
    Bring an existing reservation into line, moving stock only as needed.

    **New dates or a new room:** give back the old nights, take the new ones.
    Release first, then book — the other order refuses a one-night extension on
    a full hotel, because the guest's own room is still counted against the
    nights they are keeping. If the new nights are not there the old ones stay
    released: the guest is no longer staying on them.

    **Same nights:** pending → confirmed takes the room, confirmed → pending
    gives it back, and an overbooked stay whose room has since come free takes
    it now.

    Returns the patch actually written and whether the stay now holds stock.
    Writes nothing when there is nothing to change — a replay is a no-op.
    """
    db = get_db()
    held = _stock_booked(existing)
    moved = any(k in patch for k in ("room_type_id", "check_in", "check_out"))
    room = patch.get("room_type_id", existing["room_type_id"])
    ci = patch.get("check_in", str(existing["check_in"]))
    co = patch.get("check_out", str(existing["check_out"]))

    booked = held
    if moved:
        if held:
            booking.release_stock(db, existing["room_type_id"], existing["check_in"],
                                  existing["check_out"], int(existing.get("units") or 1))
        booked = take_stock and booking.book_stock(db, room, ci, co, 1)
    elif take_stock and not held:
        booked = booking.book_stock(db, room, ci, co, 1)
    elif not take_stock and held:
        booking.release_stock(db, room, ci, co, 1)
        booked = False

    hostos = _hostos(existing)
    note = note_for(booked)
    patch = dict(patch)
    if note != hostos.get("note"):
        # Only when *our* note changes, so a line the desk added to the
        # reservation survives every channel edit that does not change ours.
        patch["notes"] = note
    if not patch and booked == held:
        return {}, booked

    payload = patch.get("external_payload") or {
        k: v for k, v in (existing.get("external_payload") or {}).items() if k != "hostos"}
    patch["external_payload"] = {**payload, "hostos": {"stockBooked": booked, "note": note}}
    db.table("reservations").update(patch).eq("id", existing["id"]).execute()
    if moved or "total_paise" in patch:
        _write_nights(db, existing["id"], ci, co, total_paise, replace=True)
    return patch, booked


def _rebook(row: dict) -> None:
    """Undo a release of stock this reservation never took. See `_cancel`."""
    booking.book_stock(get_db(), row["room_type_id"], row["check_in"], row["check_out"],
                       int(row.get("units") or 1))


# ── side effects that must not fail the booking ──────────────────────────────

async def _push(property_id: str, room_type_id: str | None, check_in, check_out) -> None:
    """Tell the other channels. Never fatal: the nightly resync carries a miss."""
    if not room_type_id:
        return
    try:
        from app.services.repull import ari_push
        await ari_push.enqueue_ari_push(property_id, room_type_id, check_in, check_out)
    except Exception:  # noqa: BLE001 — the reservation is already right
        logger.warning("could not queue an ARI push for room type %s", room_type_id,
                       exc_info=True)


async def _escalate_overbooking(property_id: str, ctx: dict, reservation: dict,
                                source: str, guest_name: str) -> None:
    """
    Tell the hotel's team a channel sold a room HostOS did not have.

    Never fatal, and after the insert: the reservation and its note are the
    durable record, and an escalation that failed to write must not make Repull
    retry a booking that is already stored — the retry would find the row and
    escalate nothing anyway.
    """
    try:
        await escalation.raise_escalation(
            property_id=property_id,
            partner_id=ctx["partner_id"],
            kind="other",
            facts={
                "guest_name": guest_name,
                "channel": _channel_name(source),
                "check_in": str(reservation["check_in"]),
                "check_out": str(reservation["check_out"]),
                "room_type_id": reservation["room_type_id"],
                "room_name": ctx.get("room_name"),
                "units": 1,
                "booking_ref": reservation.get("external_ref") or reservation.get("ref"),
            },
            reason=(
                f"Overbooking: {_channel_name(source)} confirmed this booking but there is "
                f"no {ctx.get('room_name') or 'room'} left in HostOS for these nights."
            ),
            hotel_name=ctx["hotel"],
            reservation_id=reservation.get("id"),
        )
    except Exception:  # noqa: BLE001 — the note on the reservation still says it
        logger.exception("could not raise the overbooking escalation for reservation %s",
                         reservation.get("id"))


# ── the entry points ─────────────────────────────────────────────────────────

async def _cancel(property_id: str, existing: dict, source: str, reason: str | None) -> dict:
    """
    Cancel through HostOS's own path, which is idempotent and releases stock.

    `cancel()` releases whenever the row is `confirmed`. An overbooked stay is
    confirmed and never took stock, so that release gives back a room nobody
    returned; booking it straight back makes the pair a no-op on the counter.
    """
    already = existing["status"] == "cancelled"
    if existing["status"] not in cancellation.CANCELLABLE and not already:
        # Checked in or out: the desk's decision, not the channel's.
        logger.warning("channel cancelled reservation %s, which is %s in HostOS; left as is",
                       existing["id"], existing["status"])
        return {"outcome": "ignored", "reservation": existing}
    result = await cancellation.cancel(
        property_id, existing["id"], actor="channel", actor_name=_channel_name(source),
        reason=reason or f"Cancelled on {_channel_name(source)}",
    )
    if result.get("inventoryReleased") and not _stock_booked(existing):
        await run_db(_rebook, existing)
    if not already:
        await _push(property_id, existing.get("room_type_id"),
                    existing["check_in"], existing["check_out"])
    return {"outcome": "unchanged" if already else "cancelled", "reservation": existing}


async def upsert_external_reservation(property_id: str, obj: dict, *,
                                      source_override: str | None = None,
                                      cancelled: bool = False,
                                      push: bool = True) -> dict:
    """
    Make HostOS agree with one Repull reservation. Safe to call any number of
    times with the same object.

    Returns `{"outcome": created | updated | cancelled | unchanged | ignored,
    "reservation": row | None}`. `ignored` covers a listing no room type is
    mapped to, and a cancellation of a booking HostOS never stored — neither is
    an error, and Repull should not retry them.

    `source_override` names where the booking came from when that is not the
    channel — a PMS the hotel is importing from. `push=False` leaves the ARI
    push to the caller, so an import pushes each room type once rather than
    once per booking.
    """
    external_id = str(obj.get("id") or "")
    if not external_id:
        raise ValueError("a Repull reservation without an id")
    source = (source_override or _platform(obj)).lower()
    status_ = str(obj.get("status") or "confirmed").lower()
    existing = await run_db(_existing, property_id, external_id)

    if cancelled or status_ == "cancelled":
        if not existing:
            logger.info("cancellation for Repull reservation %s, which HostOS never stored",
                        external_id)
            return {"outcome": "ignored", "reservation": None}
        return await _cancel(property_id, existing, source, obj.get("statusDetail"))

    mapping = await connections.map_for_listing(obj.get("listingId"))
    if not mapping or str(mapping.get("property_id")) != str(property_id):
        logger.info("Repull reservation %s is on listing %s, which no room type here is "
                    "mapped to; ignored", external_id, obj.get("listingId"))
        return {"outcome": "ignored", "reservation": existing}

    room_type_id = mapping["room_type_id"]
    ctx = await run_db(_context, property_id, room_type_id)
    rate_plan_id = mapping.get("rate_plan_id") or connections._public_plan(ctx["plans"],
                                                                           room_type_id)
    if not rate_plan_id:
        # `reservations.rate_plan_id` is NOT NULL. Raised, not ignored, so the
        # event shows as failed and Repull's retry lands once a plan exists.
        raise RuntimeError(f"room type {room_type_id} has no rate plan to record "
                           f"Repull reservation {external_id} against")

    check_in = date.fromisoformat(str(obj["checkIn"])[:10])
    check_out = date.fromisoformat(str(obj["checkOut"])[:10])
    take_stock = status_ in _TAKES_STOCK
    total, currency = _money(obj)
    total_paise = round(total * 100) if (total is not None and currency == "INR") else 0
    name, phone, email = _guest(obj)
    occupancy = obj.get("occupancy") or {}

    if existing and existing["status"] == "cancelled":
        # The channel says it is live again. Reinstating would need stock taken
        # back on nights that may since have sold; that is a person's call.
        logger.warning("Repull reservation %s is %s but cancelled in HostOS; left as is",
                       external_id, status_)
        return {"outcome": "ignored", "reservation": existing}

    fields = {
        "room_type_id": room_type_id,
        "rate_plan_id": rate_plan_id,
        "guest_name": name,
        "guest_phone": phone,
        "guest_email": email,
        "check_in": check_in.isoformat(),
        "check_out": check_out.isoformat(),
        "nights": (check_out - check_in).days,
        "adults": int(occupancy.get("adults") or 1),
        "children": int(occupancy.get("children") or 0),
        "gross_paise": total_paise,
        "total_paise": total_paise,
        "status": "confirmed" if take_stock else "pending",
        "external_ref": obj.get("confirmationCode"),
        "external_payload": _trimmed(obj),
    }

    def note_for(booked: bool) -> str | None:
        return _notes(source, total, currency, take_stock and not booked, ctx["room_name"])

    if not existing:
        row = {
            **fields,
            "property_id": property_id,
            "partner_id": ctx["partner_id"],
            "units": 1,
            "discount_paise": 0,
            "tax_paise": 0,
            "paid_paise": 0,
            "payment_status": "unpaid",
            "source": source,
            # `reservations_created_by_check` is a closed set; "import" is its
            # word for a booking that did not come through HostOS's own
            # booking path, which is exactly what a channel booking is.
            "created_by": "import",
            "external_source": source,
            "external_id": external_id,
        }
        reservation = await run_db(_write_new, property_id, row, take_stock, total_paise,
                                   note_for)
        booked = _stock_booked(reservation)
        if take_stock and not booked:
            await _escalate_overbooking(property_id, ctx, reservation, source, name)
        if push and booked:
            await _push(property_id, room_type_id, check_in, check_out)
        return {"outcome": "created", "reservation": reservation}

    if existing["status"] not in _EDITABLE:
        # Arrived or left: only what the channel knows better than the desk.
        fields = {k: fields[k] for k in ("external_ref", "external_payload")}
        take_stock = _stock_booked(existing)

    held = _stock_booked(existing)
    payload_now = {k: v for k, v in (existing.get("external_payload") or {}).items()
                   if k != "hostos"}
    patch = {}
    for key, value in fields.items():
        current = payload_now if key == "external_payload" else existing.get(key)
        if key in ("check_in", "check_out"):
            current = str(current)[:10] if current is not None else None
        if current != value:
            patch[key] = value

    was_pending = existing["status"] == "pending"
    written, booked = await run_db(_update, existing, patch, take_stock, total_paise, note_for)
    if not written:
        return {"outcome": "unchanged", "reservation": existing}
    updated = {**existing, **written}
    moved = any(k in written for k in ("room_type_id", "check_in", "check_out"))

    # Escalate when this event is what made it an overbooking — a move onto
    # nights that are full, or a request the channel just confirmed. Not on a
    # later edit of a stay that was already overbooked: one ticket per
    # problem, the rule `escalation.dedupe_key` keeps for guests.
    if take_stock and not booked and (moved or was_pending):
        await _escalate_overbooking(property_id, ctx, updated, source, name)

    if push and (moved or booked != held):
        if moved:
            await _push(property_id, existing["room_type_id"],
                        existing["check_in"], existing["check_out"])
        await _push(property_id, room_type_id, check_in, check_out)
    return {"outcome": "updated", "reservation": updated}


async def handle_event(property_id: str, envelope: dict) -> str:
    """
    One `reservation.*` webhook. Returns processed | ignored; raises on a
    failure Repull should retry.

    `data.object` is the reservation. A sparse one — some events carry only
    the id and the change — is fetched whole from Repull first, so every path
    below works from the same complete shape.
    """
    if not repull_client.enabled():
        return "ignored"
    event = envelope.get("event") or ""
    data = envelope.get("data") or {}
    obj = data.get("object") or {}
    if not obj.get("id"):
        logger.warning("Repull %s without a reservation id; ignored", event)
        return "ignored"

    is_cancel = event == "reservation.cancelled" or str(obj.get("status")).lower() == "cancelled"
    if not is_cancel and not (obj.get("listingId") and obj.get("checkIn") and obj.get("checkOut")):
        from repull.api.reservations import get_reservation

        conn = await connections.connection(property_id)
        obj = await call(get_reservation.asyncio_detailed,
                         workspace_id=(conn or {}).get("workspace_id"), id=int(obj["id"]))

    # A booking that came from a PMS the hotel connected is recorded under that
    # PMS's name, the same name the import used for it.
    provider = str(((envelope.get("account") or {}).get("provider")) or "").lower()
    override = provider if provider in connections.PMS_PROVIDERS else None

    result = await upsert_external_reservation(property_id, obj, source_override=override,
                                               cancelled=event == "reservation.cancelled")
    return "ignored" if result["outcome"] == "ignored" else "processed"
