"""
Quotes, holds and reservations — the write path for selling a room.

The sequence is the whole design:

    quote  ──► hold_inventory()   stock is claimed BEFORE the guest is told a price
           ──► quotes row         immutable, expiring, with the per-night trail
    book   ──► convert_hold()     units_held → units_booked, atomically
           ──► reservations row
           ──► folio token entry

Why the hold comes first: between "₹11,400, shall I hold it?" and the guest
paying, that room has to be unsellable, or the promise is a guess. And why the
quote is immutable: a guest disputing an invoice six weeks later is asking what
they were told, and `nightly_paise` is the only honest answer.

Compensation rather than transactions. Supabase's REST interface gives no
cross-statement transaction, so every failure path after a stock move explicitly
undoes it. A room held by a request that then failed is a room nobody can sell
and nobody knows why — worse than the error the caller was going to see anyway.
"""
import logging
import random
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from fastapi import HTTPException, status

from app.db.client import get_db, run_db
from app.services.pms import negotiation
from app.services.pms import fees
from app.services.pms.availability import nightly_for_plan
from app.services.pms.pricing import (
    NightlyRate,
    PricedStay,
    nights_between,
    price_stay,
)

logger = logging.getLogger(__name__)

# Matches the frontend's hold countdown and the quote TTL. A quote outliving its
# hold would promise a room that has already been released.
HOLD_TTL_SECONDS = 15 * 60

# Ambiguous characters removed: a ref is read aloud over a phone and typed into
# a web check-in form by someone who has just got off a flight.
_REF_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# `quotes.issued_by` and `inventory_holds.source` enumerate the same actors
# under two spellings. Mapping them here rather than collapsing everything that
# is not the AI to 'employee' matters because the hold column exists to answer
# "who took this room off sale" — and a room held by the booking engine that
# reads as held by the front desk is the one question it cannot answer.
_HOLD_SOURCE = {
    "ai": "ai",
    "booking_engine": "booking_engine",
    "agent_portal": "b2b",
}


def _new_ref() -> str:
    return "TB-" + "".join(random.choices(_REF_ALPHABET, k=6))


@dataclass
class QuoteResult:
    id: str
    hold_id: str | None
    priced: PricedStay
    expires_at: datetime
    room_type_id: str
    rate_plan_id: str
    check_in: date
    check_out: date
    adults: int
    children: int


def _load_context(property_id: str, rate_plan_id: str):
    db = get_db()
    prop = (
        db.table("properties")
        .select("id,partner_id,settings,tax_inclusive,status")
        .eq("id", property_id)
        .limit(1)
        .execute()
    )
    plan = (
        db.table("rate_plans")
        .select("id,room_type_id,property_id,audience,is_active")
        .eq("id", rate_plan_id)
        .limit(1)
        .execute()
    )
    return (prop.data[0] if prop.data else None), (plan.data[0] if plan.data else None)


async def create_quote(
    *,
    property_id: str,
    rate_plan_id: str,
    check_in: date,
    check_out: date,
    units: int,
    adults: int,
    children: int,
    requested_discount_pct: int = 0,
    discount_reason: str | None = None,
    issued_by: str = "employee",
    issued_by_member: str | None = None,
    conversation_id: str | None = None,
    authorised_discount_pct: int | None = None,
    audience: str | None = None,
) -> QuoteResult:
    """
    Price a stay, hold the stock, and record the quote.

    The discount is a *request*. It is clamped against the property's floor here,
    server-side, which is what makes it safe for an AI agent to negotiate: the
    worst outcome is the floor, and the guest is never shown a number the
    property did not authorise.

    `authorised_discount_pct` raises that floor for one quote, and is how a
    manager's approval reaches the price. It does not remove the clamp — the
    ceiling becomes `properties.settings.max_approved_discount_pct`, which is
    still the property's number and still enforced here. So an approval row
    carrying 90 produces a quote at the property's approval ceiling, not at 90.
    That second clamp is not redundancy for its own sake: the first layer is an
    argument on a row somebody pressed Approve on, and this is the layer that
    does not depend on them having read it.

    `audience` is who is being quoted, and it mirrors `stay_options(audience=…)`
    exactly: the caller knows who is asking and the service enforces it. `None`
    means "any", which is right for the front desk — it books on behalf of
    travel agents and must be able to quote their rate.

    **It was not here, and its absence was a real hole.** `stay_options` has
    filtered by audience since it was written, so a guest was never *shown* a
    b2b rate plan — but nothing stopped one being *priced* if its id arrived from
    somewhere else. The whole argument for the tool layer is that a model
    guessing a UUID is not the thing standing between a guest and an agent rate.
    Found by `scripts/sales_e2e.py` calling `get_quote` with the b2b plan id
    directly, which is what a prompt injection would do if it could read one.
    """
    nights = nights_between(check_in, check_out)
    if nights <= 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Check-out must be after check-in")
    if nights > 90:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A single stay cannot exceed 90 nights")
    if units < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one room is required")

    prop, plan = await run_db(_load_context, property_id, rate_plan_id)
    if not prop or not plan or plan["property_id"] != property_id or not plan["is_active"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rate plan not found")
    if audience is not None and plan["audience"] != audience:
        # 404 rather than 403, deliberately, and the same answer the caller gets
        # for a plan id that never existed. "That rate exists but is not for you"
        # tells a guest there is a cheaper price and invites them to go looking
        # for its id.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rate plan not found")
    if prop["status"] != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, "This property is not taking bookings")

    # Prices come from the calendar, never from the caller. This is the line
    # that stops a model-generated number reaching a guest.
    #
    # `units` is passed because volume pricing is a *rate*, not a discount — see
    # the header of services/pms/rates.py. A group tier lowers the nightly here,
    # which means it flows into gross_paise, into nightly_paise, and into the GST
    # slab decision, and leaves the negotiation allowance untouched on top of it.
    resolved = await nightly_for_plan(property_id, rate_plan_id, check_in, check_out, units=units)
    if resolved is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "These dates are closed on this rate plan")
    nightly, group_tier = resolved

    policy = negotiation.resolve(prop.get("settings"))

    # The ceiling is the one binding *this rate plan*, which since
    # `negotiation_rules` may be tighter or looser than the hotel-wide setting.
    # A hotel that has configured nothing gets `rules_from_settings`, which is
    # the hotel-wide number it already had — no behaviour changes for anybody
    # who has not opted in.
    rules = await negotiation.rules_for(
        property_id,
        rate_plan_id=rate_plan_id,
        room_type_id=plan["room_type_id"],
        settings=prop.get("settings"),
    )
    max_discount = rules.approval_pct if authorised_discount_pct is not None else rules.auto_pct
    if authorised_discount_pct is not None:
        max_discount = min(max_discount, max(int(authorised_discount_pct), 0))

    # And then the floor, which is not a percentage and therefore cannot be
    # enforced by the percentage clamp alone.
    #
    # This is the third layer, and the one that matters most: the first is the
    # argument on the approval row, the second is the property's approval
    # ceiling, and this is the absolute price a hotelier typed in specifically so
    # that no ladder of percentages could ever get under it. Truncating rather
    # than rounding is deliberate — a rounded percentage can land a rupee below
    # the floor, and a floor that is breached by a rupee is not a floor.
    gross_preview = sum(p for _, p in nightly) * units
    floor_total = negotiation.floor_total_for(gross_preview, rules, nights, units)
    if floor_total is not None and gross_preview > 0:
        max_discount = min(max_discount, max(int((gross_preview - floor_total) * 100 // gross_preview), 0))

    token_pct = policy.token_pct

    priced = price_stay(
        [NightlyRate(date=d, price_paise=p) for d, p in nightly],
        units=units,
        requested_discount_pct=requested_discount_pct,
        max_discount_pct=max_discount,
        token_pct=token_pct,
        tax_inclusive=bool(prop.get("tax_inclusive")),
    )

    if priced.clamped_from_pct is not None:
        logger.info(
            "quote discount clamped property=%s asked=%s applied=%s issued_by=%s",
            property_id, priced.clamped_from_pct, priced.discount_pct_applied, issued_by,
        )

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=HOLD_TTL_SECONDS)

    def _write():
        db = get_db()
        hold_id = db.rpc(
            "hold_inventory",
            {
                "p_room_type": plan["room_type_id"],
                "p_check_in": check_in.isoformat(),
                "p_check_out": check_out.isoformat(),
                "p_units": units,
                "p_ttl_seconds": HOLD_TTL_SECONDS,
                "p_rate_plan": rate_plan_id,
                "p_source": _HOLD_SOURCE.get(issued_by, "employee"),
                "p_conversation": conversation_id,
            },
        ).execute().data

        if not hold_id:
            return None, None

        try:
            row = (
                db.table("quotes")
                .insert(
                    {
                        "property_id": property_id,
                        "room_type_id": plan["room_type_id"],
                        "rate_plan_id": rate_plan_id,
                        "partner_id": prop["partner_id"],
                        "hold_id": hold_id,
                        "check_in": check_in.isoformat(),
                        "check_out": check_out.isoformat(),
                        "units": units,
                        "adults": adults,
                        "children": children,
                        # Both the tariff the guest was shown and the money that
                        # tariff resolves to, per night. The second is what the
                        # night audit posts, so it is decided here — once, by the
                        # pricing service — rather than re-derived later from a
                        # total and a night count. See the header of
                        # 20260814_hotel_os_6_night_audit.sql.
                        "nightly_paise": [
                            {
                                "date": n.date.isoformat(),
                                "price_paise": n.gross_paise // units,
                                "room_paise": n.room_paise,
                                "tax_paise": n.tax_paise,
                            }
                            for n in priced.nights
                        ],
                        "gross_paise": priced.gross_paise,
                        "discount_paise": priced.discount_paise,
                        "discount_reason": discount_reason if priced.discount_paise else None,
                        "tax_paise": priced.tax_paise,
                        "total_paise": priced.total_paise,
                        "token_paise": priced.token_paise,
                        "balance_paise": priced.balance_paise,
                        "issued_by": issued_by,
                        "issued_by_member": issued_by_member,
                        "conversation_id": conversation_id,
                        "expires_at": expires_at.isoformat(),
                        # Which of the hotel's own volume bands produced this
                        # price, when one did. Null on an ordinary quote.
                        "group_tier_id": group_tier.tier_id if group_tier else None,
                    }
                )
                .execute()
            )
        except Exception:
            # The stock is already held. Failing here without releasing it would
            # take the room off sale for 15 minutes for no reason.
            db.rpc("release_hold", {"p_hold": hold_id}).execute()
            raise

        return row.data[0], hold_id

    quote_row, hold_id = await run_db(_write)
    if quote_row is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Those dates are no longer available in the number of rooms requested",
        )

    return QuoteResult(
        id=quote_row["id"],
        hold_id=hold_id,
        priced=priced,
        expires_at=expires_at,
        room_type_id=plan["room_type_id"],
        rate_plan_id=rate_plan_id,
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        children=children,
    )


def _release(hold_id: str) -> None:
    get_db().rpc("release_hold", {"p_hold": hold_id}).execute()


async def abandon_quote(quote_id: str, hold_id: str | None) -> None:
    """
    Give back a quote that will never be booked, and the room it was holding.

    Used where a caller has priced a stay and *then* found a reason to refuse it
    — the agent portal's credit check is the first such caller. Leaving the hold
    to expire on its own would be correct and would also take a room off sale
    for fifteen minutes for a booking everybody already knows is not happening.

    Deliberately silent about a hold that no longer exists, for the same reason
    `release_hold` is: cleanup after a refusal must never be the thing that
    fails.
    """
    if hold_id:
        await run_db(_release, hold_id)

    def _mark():
        get_db().table("quotes").update({"status": "superseded"}).eq("id", quote_id).execute()

    await run_db(_mark)


def _payment_status(paid: int, total: int) -> str:
    if paid <= 0:
        return "unpaid"
    if paid >= total:
        return "paid"
    # A token is a deposit against a confirmed booking; a part payment is
    # someone paying down a balance. The distinction drives what the desk
    # chases, so it is worth keeping rather than collapsing to "partial".
    return "token_paid" if paid <= total * 0.35 else "partially_paid"



def _resolve_guest(
    db, property_id: str, partner_id: str,
    name: str, phone: str | None, email: str | None,
) -> str | None:
    """
    The person, as opposed to the stay.

    A reservation carries a name, a phone and an email as *text*. That is
    enough to run a front desk and not enough to run a hotel: "has this person
    stayed before", "how much have they spent with us", "who should get the
    returning-guest offer" are all questions about a person, and none of them
    can be asked of a string on a booking row.

    Until this existed nothing wrote `guests` on the booking path, so every
    real booking produced a stay with `guest_id` null. The guest list was
    empty, the marketing audience could not see anybody who had actually
    stayed, and the AI employee's `get_guest` never matched a returning guest.
    The rows existed; nothing ever created one.

    **Matched within the property, and only ever within it.** The same phone
    number at two hotels is two guest records. Joining them would be a
    cross-property leak dressed up as a convenience -- the same rule
    `messaging.service` follows when it resolves a conversation.

    Matched on the **last ten digits** of the phone, because a guest is
    `+919876543210` on WhatsApp, `9876543210` at the desk and `09876543210` on
    an OTA, and all three are one person. Email is the fallback for a booking
    with no phone at all.

    Never fatal. A hotel that cannot write a guest record still has to be able
    to take the booking -- the stay is the thing the guest is standing there
    for.
    """
    digits = "".join(c for c in (phone or "") if c.isdigit())
    found: list[dict] = []
    try:
        if len(digits) >= 10:
            found = (
                db.table("guests").select("id,stays_count")
                .eq("property_id", property_id)
                .ilike("phone", f"%{digits[-10:]}")
                .limit(1).execute().data or []
            )
        if not found and (email or "").strip():
            found = (
                db.table("guests").select("id,stays_count")
                .eq("property_id", property_id)
                .ilike("email", (email or "").strip())
                .limit(1).execute().data or []
            )
        if found:
            return found[0]["id"]

        return (
            db.table("guests").insert({
                "property_id": property_id,
                "partner_id": partner_id,
                "name": (name or "Guest").strip()[:120],
                "phone": (phone or "").strip() or None,
                "email": (email or "").strip() or None,
                # `stays_count` and `total_spend_paise` are deliberately left at
                # their defaults. They describe stays that have *happened*, and
                # a booking made this minute has not. Check-out is what moves
                # them, so a guest who cancels never counts as having stayed.
            }).execute().data[0]["id"]
        )
    except Exception:  # noqa: BLE001 — a guest record must not fail a booking
        logger.warning(
            "could not resolve a guest record for %s at %s", name, property_id,
            exc_info=True,
        )
        return None


# ── the stock and the row, shared by every way a booking arrives ─────────────
#
# `create_reservation` sells from a quote. A booking taken on Booking.com or
# Airbnb (services/repull/reservations.py) arrives already sold, at the
# channel's price, with no quote — but it moves the same counter and writes the
# same row, and a second spelling of either is how the counter drifts. So the
# three steps both paths take live here, once.

def book_stock(db, room_type_id: str, check_in: str, check_out: str, units: int) -> bool:
    """
    Take `units` rooms off sale for these nights. False if they are not there.

    `book_inventory` answers with what is left, or null/negative when it
    refused — and a refusal moves nothing, so there is nothing to undo.
    """
    left = db.rpc(
        "book_inventory",
        {
            "p_room_type": room_type_id,
            "p_check_in": str(check_in),
            "p_check_out": str(check_out),
            "p_units": int(units),
        },
    ).execute().data
    return not (left is None or left < 0)


def release_stock(db, room_type_id: str, check_in: str, check_out: str, units: int) -> None:
    """Put rooms back on sale. The same RPC cancellation releases through."""
    db.rpc(
        "release_booking",
        {
            "p_room_type": room_type_id,
            "p_check_in": str(check_in),
            "p_check_out": str(check_out),
            "p_units": int(units),
        },
    ).execute()


def insert_reservation(db, row: dict, *, compensate) -> dict:
    """
    Insert one reservation under a fresh booking reference.

    A ref collision is a 1-in-30-million coincidence and a unique constraint
    violation; retrying twice is cheaper than a wider ref. Any *other* failure
    means the stock the caller already moved belongs to nobody, so `compensate`
    runs before the error goes up — see the module docstring on compensation.
    """
    for _ in range(3):
        try:
            return (
                db.table("reservations")
                .insert({"ref": _new_ref(), **row})
                .execute()
            ).data[0]
        except Exception as exc:  # noqa: BLE001 — retried, then re-raised
            if "reservations_ref_key" not in str(exc):
                compensate()
                raise
    compensate()
    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not allocate a booking reference")


async def create_reservation(
    *,
    property_id: str,
    quote_id: str,
    guest_name: str,
    guest_phone: str | None,
    guest_email: str | None,
    source: str,
    token_paise: int,
    payment_method: str | None,
    notes: str | None,
    created_by: str,
    member_id: str | None,
    member_name: str | None,
    conversation_id: str | None = None,
) -> dict:
    """
    Turn a quote into a reservation.

    Consuming the quote is what stops a double-submit selling the same held room
    twice: the second call finds it already converted and is refused.
    """

    def _write():
        db = get_db()

        quote = (
            db.table("quotes")
            .select("*")
            .eq("id", quote_id)
            .eq("property_id", property_id)
            .limit(1)
            .execute()
        )
        if not quote.data:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Quote not found")
        q = quote.data[0]

        if q["status"] != "active":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "This quote has already been used. Request a new price.",
            )
        if datetime.fromisoformat(q["expires_at"].replace("Z", "+00:00")) < datetime.now(timezone.utc):
            raise HTTPException(status.HTTP_409_CONFLICT, "This quote has expired. Request a new price.")

        check_in = date.fromisoformat(q["check_in"])
        check_out = date.fromisoformat(q["check_out"])

        # Convert the hold first. If it fails the stock was never ours and no
        # reservation should exist.
        if q.get("hold_id"):
            converted = db.rpc("convert_hold", {"p_hold": q["hold_id"], "p_reservation": None}).execute().data
            if not converted:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "The hold on these rooms expired. Request a new price.",
                )
        elif not book_stock(db, q["room_type_id"], q["check_in"], q["check_out"], q["units"]):
            raise HTTPException(status.HTTP_409_CONFLICT, "Those dates are no longer available")

        paid = max(0, min(int(token_paise or 0), q["total_paise"]))

        # What StayOS keeps and what the hotel is owed — decided here, once, and
        # never recomputed. §37: a booking preserves its own financial truth, so
        # renegotiating this partner's rate next year must not change what this
        # stay was worth. A partner row that cannot be read falls back to the
        # standard rate rather than to no fee; see services/pms/fees.py.
        partner_row = (
            db.table("partners").select("platform_fee_bps")
            .eq("id", q["partner_id"]).limit(1).execute().data or []
        )
        fee_columns = fees.snapshot_columns(
            q["total_paise"], partner_row[0].get("platform_fee_bps") if partner_row else None,
        )

        # Before the reservation, so the row is linked from the moment it
        # exists rather than by a second write that can fail on its own.
        guest_id = _resolve_guest(
            db, property_id, q["partner_id"], guest_name, guest_phone, guest_email,
        )

        def _compensate():
            # The stock moved, the reservation did not. Both branches above
            # moved it — `convert_hold` and `book_inventory` each end in
            # `units_booked` — so both are given back. This used to release
            # only after a converted hold, which left a failed hold-less insert
            # holding a room nobody had booked.
            release_stock(db, q["room_type_id"], q["check_in"], q["check_out"], q["units"])

        reservation = insert_reservation(
            db,
            {
                "property_id": property_id,
                "partner_id": q["partner_id"],
                "room_type_id": q["room_type_id"],
                "rate_plan_id": q["rate_plan_id"],
                "quote_id": q["id"],
                "guest_name": guest_name,
                "guest_phone": guest_phone,
                "guest_email": guest_email,
                "guest_id": guest_id,
                "check_in": q["check_in"],
                "check_out": q["check_out"],
                "nights": (check_out - check_in).days,
                "units": q["units"],
                "adults": q["adults"],
                "children": q["children"],
                "gross_paise": q["gross_paise"],
                "discount_paise": q["discount_paise"],
                "tax_paise": q["tax_paise"],
                "total_paise": q["total_paise"],
                "paid_paise": paid,
                # Read off the quote, never off the request. There
                # is no `agencyId` parameter anywhere on the booking
                # path, so the only way a stay can be billed to a
                # credit line is if `b2b_commit_credit` put the
                # agency on the quote — which it only does under the
                # agency's own row lock, having checked the limit.
                "b2b_agent_id": q.get("b2b_agent_id"),
                "source": source,
                "status": "confirmed",
                "payment_status": _payment_status(paid, q["total_paise"]),
                "notes": notes,
                "created_by": created_by,
                # The thread it was sold in, if it was sold in one.
                # Falls back to the quote's own thread: an order
                # raised on the booking page for a quote the AI
                # made still belongs to that conversation, and the
                # guest asking about it is on that conversation.
                "conversation_id": conversation_id or q.get("conversation_id"),
                **fee_columns,
            },
            compensate=_compensate,
        )
        db.table("quotes").update({"status": "converted"}).eq("id", q["id"]).execute()

        # The per-night trail. This is what the night audit posts from, which is
        # why the money is copied off the quote rather than recomputed: the
        # guest agreed to these numbers, and a stay whose folio is rebuilt from
        # a total divided by a night count will not sum back to them.
        nights_rows = []
        for n in q.get("nightly_paise") or []:
            if "room_paise" not in n:
                # A quote issued before per-night money existed. Quotes live 15
                # minutes, so this is all but unreachable — and refusing is far
                # better than guessing, because the guess becomes a charge.
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "This quote was issued under older pricing. Please re-check availability.",
                )
            nights_rows.append(
                {
                    "reservation_id": reservation["id"],
                    "date": n["date"],
                    "rate_paise": n["price_paise"],
                    "room_paise": n["room_paise"],
                    "tax_paise": n["tax_paise"],
                }
            )
        if nights_rows:
            db.table("reservation_nights").insert(nights_rows).execute()

        if paid > 0:
            db.table("folio_entries").insert(
                {
                    "reservation_id": reservation["id"],
                    "property_id": property_id,
                    "partner_id": q["partner_id"],
                    "kind": "payment",
                    "description": "Booking token" if paid < q["total_paise"] else "Payment received",
                    # Payments are stored negative. The sign is applied once,
                    # here, so no caller has to remember which way round it goes.
                    "amount_paise": -paid,
                    "method": payment_method or "cash",
                    "posted_by_member": member_id,
                    "posted_by_name": member_name or "Front desk",
                    "idempotency_key": f"token:{reservation['id']}",
                }
            ).execute()

        return reservation

    reservation = await run_db(_write)
    # The stock moved; tell the channels. Never raises — a sale must not fail
    # because Repull is unreachable.
    from app.services.repull import ari_push
    await ari_push.enqueue_stay(property_id, reservation.get("room_type_id"),
                                reservation.get("check_in"), reservation.get("check_out"))
    return reservation
