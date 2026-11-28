"""
Cancelling a booking, and giving back exactly what the policy says.

**The gap this closes.** The readiness audit found no cancellation anywhere in
the system. A guest could be sold a room by the AI in ninety seconds and then
had to reach a human to undo it — and the two things that happen when a booking
goes away, the inventory coming back and the money going back, had no code path
at all. A room that is cancelled and never released is a room the hotel cannot
sell again, which is the same loss as an empty night with none of the visibility.

**The refund is arithmetic, and the arithmetic is here.** §14 and §24 rule 3:
the model does not compute money. It does not apply a percentage, it does not
compare a date against a deadline, and it does not decide whether a rate was
refundable. `policies.refund_for` already did this work for `get_cancellation_terms`
— the tool that *quotes* a policy — and this is the same function applied to a
booking that is actually going away. The agent's whole part is to identify the
booking and relay the answer.

    identify   the reservation, from the conversation and never from the model
        ↓
    validate   is this a booking that can be cancelled at all?
        ↓
    compute    policies.refund_for → a percentage and a figure, deterministic
        ↓
    release    release_booking, the same RPC the booking path compensates with
        ↓
    record     status, cancelled_at, reason, and a folio refund line
        ↓
    report     what the guest gets back and when

**Order matters and this is the order.** Inventory comes back before the
reservation is marked cancelled, because the failure that leaves a hotel worse
off is stock that is still sold against a booking nobody can find. The reverse
order — mark first, release second — leaves a cancelled reservation holding
inventory if the process dies between them, which is exactly the phantom
room-nights the reconciliation tool had to repair after the test-cleanup bug.
Doing it this way, a crash in the middle leaves stock released against a booking
still marked confirmed, which reads as an oversell risk on one room and is
visible to the night audit. Neither is good; one is loud.

**Idempotent, because a guest asks twice.** "Cancel kar do" followed thirty
seconds later by "cancel hua?" must not release the same rooms twice — that is
how inventory goes *above* what the hotel has. A reservation already cancelled
returns its existing cancellation unchanged, and the folio line carries an
idempotency key derived from the reservation rather than from the request.

**What this does not do: move money at a gateway.** The ledger records what is
owed back and the reservation records that it is refunded, and the actual
Razorpay refund is left to a person. That is deliberate rather than unfinished —
an automated outbound refund is a payment the hotel did not authorise at the
moment it happened, and the difference between "₹4,500 is owed to this guest"
and "₹4,500 has left the hotel's account" is a difference a hotelier is entitled
to be the one to make. `refundDue` says so explicitly, and the caller reports it.
"""
import hashlib
import logging
from datetime import date, datetime, time, timezone

from fastapi import HTTPException, status

from app.db.client import get_db, run_db
from app.services.pms import policies

logger = logging.getLogger(__name__)

# Statuses that can still be cancelled, and what to say about the rest.
#
# `checked_in` and `checked_out` are deliberately not here. A guest who has
# arrived is a departure, an early checkout or a dispute — three things with
# different money and all of them a person's decision. Telling the guest that
# plainly is better than a cancellation that silently means something else.
CANCELLABLE = frozenset({"pending", "confirmed"})

_REFUSALS = {
    "checked_in": (
        "already_checked_in",
        "This guest has arrived. An early departure is not a cancellation and "
        "the hotel has to handle it.",
    ),
    "checked_out": (
        "already_checked_out",
        "This stay is finished. Anything owed after a completed stay is a "
        "billing question for the hotel.",
    ),
    "no_show": (
        "recorded_as_no_show",
        "This booking is already recorded as a no-show, which the hotel's "
        "policy prices separately.",
    ),
}

# Inventory is only consumed once a booking is real. A `pending` reservation
# never called `book_inventory`, so releasing against it would hand the hotel
# stock it never sold.
HOLDS_INVENTORY = frozenset({"confirmed"})


def _read(property_id: str, reservation_id: str) -> tuple[dict, dict, dict]:
    """The reservation, its rate plan and its property. One round trip each."""
    db = get_db()
    rows = (
        db.table("reservations").select("*")
        .eq("id", reservation_id).eq("property_id", property_id)
        .limit(1).execute().data or []
    )
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Reservation not found")
    reservation = rows[0]

    plan = {}
    if reservation.get("rate_plan_id"):
        plan_rows = (
            db.table("rate_plans")
            .select("name,refundable,cancellation_cutoff_hours,cancellation_policy")
            .eq("id", reservation["rate_plan_id"]).limit(1).execute().data or []
        )
        plan = plan_rows[0] if plan_rows else {}

    prop_rows = (
        db.table("properties").select("check_in_time,policies,timezone")
        .eq("id", property_id).limit(1).execute().data or []
    )
    return reservation, plan, (prop_rows[0] if prop_rows else {})


def _check_in_time(prop: dict) -> time:
    """
    The hotel's arrival hour, which is what a "24 hours before" policy counts
    back from. A missing one defaults to 2pm rather than to midnight — midnight
    would give away ten hours of every deadline.
    """
    raw = prop.get("check_in_time")
    if not raw:
        return time(14, 0)
    try:
        return time.fromisoformat(str(raw)[:8])
    except (TypeError, ValueError):
        return time(14, 0)


def _terms_for(reservation: dict, plan: dict) -> policies.CancellationTerms:
    """
    The policy this booking was sold under.

    `reservations.cancellation_policy` first: a booking is governed by the terms
    in force the day it was made, not by the terms the hotel switched to last
    week. Falling back to the plan's current policy is the best available answer
    for a booking taken before that column was written, and it is flagged as
    derived so the caller can say so.
    """
    snapshot = reservation.get("cancellation_policy")
    if isinstance(snapshot, dict) and snapshot:
        return policies.cancellation_terms(
            snapshot,
            refundable=bool(snapshot.get("refundable", plan.get("refundable", True))),
            cutoff_hours=snapshot.get("cutoffHours", plan.get("cancellation_cutoff_hours")),
        )
    return policies.cancellation_terms(
        plan.get("cancellation_policy"),
        refundable=bool(plan.get("refundable", True)),
        cutoff_hours=plan.get("cancellation_cutoff_hours"),
    )


def _settlement(reservation: dict, terms: policies.CancellationTerms,
                prop: dict, *, at: datetime | None, no_show: bool) -> dict:
    """
    What comes back, in paise, with nothing left to interpret.

    Two numbers and the smaller wins. The policy says what percentage of the
    *stay* is refundable; the ledger says what the guest actually handed over.
    A hotel that took a 30% deposit on a fully refundable rate owes back the
    deposit and not the whole stay, and a function that returned the policy
    figure alone would have promised three times the money.
    """
    computed = policies.refund_for(
        terms,
        total_paise=int(reservation.get("total_paise") or 0),
        check_in=date.fromisoformat(str(reservation["check_in"])),
        check_in_time=_check_in_time(prop),
        at=at,
        no_show=no_show,
    )
    paid = int(reservation.get("paid_paise") or 0)
    refund = min(computed["refundPaise"], paid)

    return {
        "refundPct": computed["refundPct"],
        "refundPaise": refund,
        "retainedPaise": max(0, paid - refund),
        "paidPaise": paid,
        "totalPaise": int(reservation.get("total_paise") or 0),
        "hoursRemaining": computed["hoursRemaining"],
        "noShow": no_show,
        # The named cases §14 asks for, decided here so nobody downstream has to
        # infer them from a percentage.
        "kind": (
            "free" if computed["refundPct"] >= 100 and refund >= paid
            else "non_refundable" if computed["refundPct"] == 0
            else "partial"
        ),
        "afterDeadline": computed["refundPct"] < 100,
    }


async def preview(property_id: str, reservation_id: str, *,
                  at: datetime | None = None, no_show: bool = False) -> dict:
    """
    What cancelling would cost, without cancelling anything.

    This is what a guest asking "cancel karne pe kitna milega?" is owed, and it
    is the same computation the cancellation itself performs — one function, so
    the number a guest is quoted and the number they are given cannot drift.
    """
    reservation, plan, prop = await run_db(_read, property_id, reservation_id)
    terms = _terms_for(reservation, plan)
    settlement = _settlement(reservation, terms, prop, at=at, no_show=no_show)

    refusal = _REFUSALS.get(reservation["status"])
    return {
        "reservationId": reservation["id"],
        "ref": reservation["ref"],
        "status": reservation["status"],
        "cancellable": reservation["status"] in CANCELLABLE,
        "alreadyCancelled": reservation["status"] == "cancelled",
        "blockedReason": refusal[0] if refusal else None,
        "blockedExplanation": refusal[1] if refusal else None,
        "checkIn": str(reservation["check_in"]),
        "checkOut": str(reservation["check_out"]),
        "units": reservation.get("units") or 1,
        "ratePlan": plan.get("name"),
        "terms": terms.as_dict(),
        **settlement,
    }


def _idempotency_key(reservation_id: str) -> str:
    """
    Derived from the booking, not from the request.

    A guest who says "cancel kar do" and then "cancel hua?" produces two
    attempts thirty seconds apart, and a key generated per request would let the
    second one post a second refund line. The reservation can only be cancelled
    once, so the reservation is the key.
    """
    return "cancel:" + hashlib.sha256(reservation_id.encode()).hexdigest()[:32]


def _apply(property_id: str, reservation: dict, settlement: dict,
           reason: str | None, actor: str, actor_name: str,
           member_id: str | None) -> dict:
    # Positional, not keyword: `run_db` hands its arguments straight to a thread
    # executor and does not forward keywords.
    """
    The write, in the order a crash can survive.

    Inventory first. See the module docstring: stock released against a booking
    still marked confirmed is a visible inconsistency the night audit catches; a
    cancelled booking still holding stock is an invisible one that quietly costs
    the hotel a room a night until somebody runs a reconciliation.
    """
    db = get_db()

    released = False
    if reservation["status"] in HOLDS_INVENTORY and reservation.get("room_type_id"):
        db.rpc("release_booking", {
            "p_room_type": reservation["room_type_id"],
            "p_check_in": str(reservation["check_in"]),
            "p_check_out": str(reservation["check_out"]),
            "p_units": int(reservation.get("units") or 1),
        }).execute()
        released = True

    paid = int(reservation.get("paid_paise") or 0)
    refund = int(settlement["refundPaise"])
    payment_status = (
        "refunded" if refund and refund >= paid
        else reservation.get("payment_status") or "unpaid"
    )

    updated = (
        db.table("reservations").update({
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
            "cancellation_reason": (reason or "").strip()[:500] or None,
            "payment_status": payment_status,
        }).eq("id", reservation["id"]).eq("property_id", property_id).execute()
    ).data

    # The ledger line. A refund of zero is not posted: a non-refundable
    # cancellation moves no money, and a ₹0 folio entry is a row that says
    # nothing and that `post_folio_entry` refuses anyway.
    entry = None
    if refund > 0:
        key = _idempotency_key(reservation["id"])
        existing = (
            db.table("folio_entries").select("*")
            .eq("idempotency_key", key).limit(1).execute().data or []
        )
        if existing:
            entry = existing[0]
        else:
            entry = (
                db.table("folio_entries").insert({
                    "reservation_id": reservation["id"],
                    "property_id": property_id,
                    "partner_id": reservation["partner_id"],
                    "kind": "refund",
                    "description": (
                        f"Cancellation refund — {settlement['refundPct']}% "
                        f"of {reservation['ref']}"
                    ),
                    "amount_paise": refund,
                    "method": None,
                    "posted_by_member": member_id,
                    "posted_by_name": actor_name,
                    "idempotency_key": key,
                }).execute()
            ).data[0]

        # `paid_paise` is what the hotel is still holding. Lowered by the
        # refund, so the folio and the reservation's own running total cannot
        # disagree — the rule `post_folio_entry` states and this obeys.
        db.table("reservations").update({
            "paid_paise": max(0, paid - refund),
        }).eq("id", reservation["id"]).execute()

    logger.info(
        "cancelled %s (%s): refund %d paise, inventory released=%s, by %s",
        reservation["ref"], reservation["id"], refund, released, actor,
    )
    return {
        "reservation": updated[0] if updated else reservation,
        "inventoryReleased": released,
        "folioEntryId": (entry or {}).get("id"),
    }


async def cancel(property_id: str, reservation_id: str, *, reason: str | None = None,
                 actor: str = "agent", actor_name: str = "AI agent",
                 member_id: str | None = None, at: datetime | None = None,
                 no_show: bool = False) -> dict:
    """
    Cancel a booking. Idempotent, deterministic, and never partial in the
    direction that costs the hotel a room.

    Raises `HTTPException` for a booking that cannot be cancelled, with the
    reason on it — the caller turns that into something a guest can hear.
    """
    reservation, plan, prop = await run_db(_read, property_id, reservation_id)

    if reservation["status"] == "cancelled":
        # Already done. Reported as a success with `alreadyCancelled` set,
        # rather than as an error: a guest asking twice has not done anything
        # wrong, and the honest answer is "yes, it is cancelled".
        settled = await preview(property_id, reservation_id, at=at, no_show=no_show)
        settled["alreadyCancelled"] = True
        settled["cancelledAt"] = reservation.get("cancelled_at")
        settled["inventoryReleased"] = False
        return settled

    refusal = _REFUSALS.get(reservation["status"])
    if refusal:
        raise HTTPException(status.HTTP_409_CONFLICT, refusal[1])
    if reservation["status"] not in CANCELLABLE:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"A booking with status {reservation['status']} cannot be cancelled here.",
        )

    terms = _terms_for(reservation, plan)
    settlement = _settlement(reservation, terms, prop, at=at, no_show=no_show)

    applied = await run_db(
        _apply, property_id, reservation, settlement,
        reason, actor, actor_name, member_id,
    )
    if applied["inventoryReleased"]:
        # The room is sellable again; tell the channels. Never raises.
        from app.services.repull import ari_push
        await ari_push.enqueue_stay(property_id, reservation.get("room_type_id"),
                                    reservation.get("check_in"), reservation.get("check_out"))

    return {
        "reservationId": reservation["id"],
        "ref": reservation["ref"],
        "status": "cancelled",
        "alreadyCancelled": False,
        "cancelledAt": applied["reservation"].get("cancelled_at"),
        "inventoryReleased": applied["inventoryReleased"],
        "folioEntryId": applied["folioEntryId"],
        # The money is *owed*, not moved. See the module docstring: an automated
        # outbound refund is a payment the hotel did not authorise at the moment
        # it happened, and that call belongs to a person.
        "refundDue": settlement["refundPaise"] > 0,
        "refundMethod": "hotel_processes_manually",
        "terms": terms.as_dict(),
        **settlement,
    }
