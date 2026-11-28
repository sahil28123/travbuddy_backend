"""
Booking.

The agent cannot conjure a reservation. `create_booking` takes a `quote_id` and
nothing else that matters — no price, no dates, no room type — because
`services/pms/booking.create_reservation()` reads all of that from the quote it
already priced and held. There is no argument for the model to get wrong and no
argument for a guest to talk it into.

That is also what makes the flow safe under retry: a quote can be converted
once. A second call finds it consumed and is refused, so a model that repeats
itself produces one booking, not two.

`release_hold` exists because a guest who says "actually, no" should not leave a
room unsellable for fifteen minutes. It is level 2 rather than 3: giving stock
back is the safe direction.
"""
from pydantic import BaseModel, Field

from app.db.client import get_db, run_db
from app.services.ai.context import AgentContext
from app.services.ai.errors import OutOfScope, ToolUnavailable
from app.services.ai.ids import ResourceId
from app.services.ai import registry
from app.services.ai.registry import tool
from app.services.ai.tools import ACTOR_RESERVATION
from app.services.pms import booking as booking_service
from app.services.pms import creators as creators_service
from app.services.pms import money


class BookingArgs(BaseModel):
    quote_id: ResourceId = Field(description="From get_quote. The price and dates come from it.")
    guest_name: str = Field(min_length=2, max_length=120)
    guest_phone: str | None = Field(default=None, max_length=20)
    guest_email: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=500)


class HoldArgs(BaseModel):
    hold_id: ResourceId = Field(description="From get_quote.")


class CancelArgs(BaseModel):
    """
    Deliberately almost empty.

    There is no reservation id, no refund amount and no percentage. The booking
    comes from the conversation — `scoped_to_conversation` and
    `ctx.allowed_reservations` — and the money comes from the policy the stay
    was sold under. A model that could pass either could be talked into
    cancelling somebody else's booking or into refunding more than the hotel
    owes, and both of those are one guest sentence away from happening.
    """

    reason: str | None = Field(
        default=None, max_length=300,
        description=(
            "Why the guest says they are cancelling, in their own words. "
            "Recorded on the booking for the hotel; it does not change the "
            "refund, which is the policy's decision and not yours."
        ),
    )


async def _revalidate_booking(args: "BookingArgs", ctx: AgentContext) -> None:
    """
    Is the quote a manager approved still the quote we would honour?

    This is the case the whole revalidation path exists for. A quote holds
    inventory for fifteen minutes and carries a price. Approve it twenty minutes
    later and the hold has expired, the room may have sold, and the price on the
    row is a promise nobody is standing behind any more.

    Consuming it anyway would sell a room the hotel no longer has at a price it
    no longer offers, on the strength of an approval that was true when it was
    given.
    """
    from datetime import datetime, timezone

    def _read():
        rows = (
            get_db().table("quotes")
            .select("id,property_id,status,expires_at,total_paise,hold_id")
            .eq("id", args.quote_id).limit(1).execute().data or []
        )
        return rows[0] if rows else None

    quote = await run_db(_read)
    if not quote or quote["property_id"] != ctx.property_id:
        raise OutOfScope(
            f"QUOTE_NOT_FOUND: quote {args.quote_id} is not this property's",
            "Something is out of step with that price — the team will follow up.",
        )

    if quote["status"] != "active":
        raise OutOfScope(
            f"QUOTE_CONSUMED: the quote is {quote['status']}, not active",
            "That price has already been used — the team will follow up.",
        )

    expires = quote.get("expires_at")
    if expires:
        when = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        if when < datetime.now(timezone.utc):
            raise OutOfScope(
                f"QUOTE_EXPIRED: the quote expired at {when.isoformat()} and "
                "its hold has been released",
                "That price has expired — the team will send you a fresh one.",
            )


async def _verify_booking(args: "BookingArgs", ctx: AgentContext,
                          result_id: str | None) -> dict:
    """
    Read the reservation back before anyone is told they have one.

    **The worst sentence this product can produce is "your rooms are booked"
    when they are not** — a guest who believes it acts on it by travelling. The
    guardrails already refuse that claim when no booking tool ran; this closes
    the other half, where the tool *did* run and the row is not there.

    Read by id off `reservations`, scoped to the property, independently of the
    write path. A booking in `cancelled` counts as unverified: the row existing
    is not the claim, the stay existing is.
    """
    from app.services.ai.errors import MutationUnverified

    if not result_id:
        raise MutationUnverified(
            "create_booking returned no reservation id, so nothing can be confirmed",
        )

    def _read():
        rows = (
            get_db().table("reservations")
            .select("id,property_id,ref,status,check_in,check_out,units")
            .eq("id", result_id).eq("property_id", ctx.property_id)
            .limit(1).execute().data or []
        )
        return rows[0] if rows else None

    stay = await run_db(_read)
    if not stay:
        raise MutationUnverified(
            f"create_booking returned reservation {result_id} and it is not on "
            f"this property",
            guest_message=(
                "I couldn't confirm that booking went through — let me get "
                "someone from the team onto it right away."
            ),
            model_message=(
                "The booking did NOT complete. Do NOT tell the guest they are "
                "booked, reserved or confirmed, and do not give a reference. "
                "Say you are checking and call request_human_handoff."
            ),
        )

    if stay.get("status") in ("cancelled", "no_show"):
        raise MutationUnverified(
            f"reservation {stay.get('ref')} reads {stay.get('status')!r} "
            f"immediately after being created",
            model_message=(
                f"The booking is '{stay.get('status')}' and the guest does NOT "
                f"have a room. Do not say it is confirmed. Call "
                f"request_human_handoff."
            ),
        )

    return {
        "ref": stay.get("ref"),
        "status": stay.get("status"),
        "checkIn": stay.get("check_in"),
        "checkOut": stay.get("check_out"),
        "rooms": stay.get("units"),
    }


@tool(
    name="create_booking",
    description=(
        "Turn a quote into a confirmed reservation. The price, dates and rooms "
        "all come from the quote — you cannot set them here. Only call this "
        "when the guest has clearly agreed to the quoted total."
    ),
    input_model=BookingArgs,
    permission="frontdesk",
    autonomy_floor=3,
    mutating=True,
    usage_type="ai.tool_booking",
    result_type="reservation",
    creates_result=True,
    revalidate=_revalidate_booking,
    # Even at level 3. The first hotel to let an AI take bookings unattended
    # will not be the first hotel to try it, and the queue costs a manager ten
    # seconds against a booking they cannot un-take.
    always_requires_approval=True,
    # And this is the sentence that keeps the gate from reading as a fault.
    #
    # Because the flag above is unconditional, *every* guest who says "book it"
    # lands here. Told only that approval was required, the model apologised —
    # "I'm sorry, I wasn't able to book the room directly" — and a guest who
    # had just been quoted a real price heard the hotel's own safety rule as
    # the hotel being unable to sell them a room.
    #
    # There is nothing to be sorry about: the guest can pay right now, and
    # paying is what confirms a booking anyway. So the deferral points at the
    # road that works instead of describing the one that is closed.
    when_deferred=(
        "Nothing is booked and no room is held. Do NOT apologise and do NOT "
        "say the booking failed — a hotel confirms on payment. Call "
        "create_booking_payment_link for this same quote and give the guest "
        "the link. Tell them the room is confirmed once the payment goes "
        "through; do NOT tell them it is reserved, held or booked before "
        "then, because it is not and somebody else may take it. Only if that "
        "also refuses should you hand over to the team."
    ),
    verify=_verify_booking,
    requires=(registry.OFF_PLATFORM,),
)
async def create_booking(args: BookingArgs, ctx: AgentContext) -> tuple[dict, str]:
    ceiling = ctx.limits.get("max_booking_value_paise")
    if ceiling is not None:
        def _read():
            rows = (
                get_db()
                .table("quotes")
                .select("total_paise,status,property_id")
                .eq("id", args.quote_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            return rows[0] if rows else None

        quote = await run_db(_read)
        if not quote or quote["property_id"] != ctx.property_id:
            raise OutOfScope(
                f"quote {args.quote_id} does not belong to this property",
                "Something's out of step with that price — let me get the team.",
            )
        if quote["total_paise"] > int(ceiling):
            raise OutOfScope(
                f"quote total {quote['total_paise']} exceeds the agent's "
                f"max_booking_value_paise of {ceiling}",
                "That's a larger booking than I can complete myself — "
                "I'll bring in the hotel team to finish it with you.",
            )

    try:
        reservation = await booking_service.create_reservation(
            property_id=ctx.property_id,
            quote_id=args.quote_id,
            guest_name=args.guest_name,
            guest_phone=args.guest_phone,
            guest_email=args.guest_email,
            source="whatsapp" if ctx.channel == "whatsapp" else ctx.channel,
            token_paise=0,
            payment_method=None,
            notes=args.notes,
            created_by=ACTOR_RESERVATION,
            member_id=None,
            member_name=ctx.principal.display_name,
        )
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(
            f"booking failed: {type(exc).__name__}: {exc}",
            "I couldn't complete that booking — putting you through to the team now.",
        ) from exc

    # Who sent this guest, decided by the server from the touch chain. The model
    # is not consulted and could not be: it has no way to name a creator, and
    # `resolve` reads the conversation id off the run's own context rather than
    # off anything the model produced. An AI that could assign attribution would
    # be an AI that could assign money.
    await creators_service.resolve(
        reservation["id"], conversation_id=ctx.conversation_id
    )

    return (
        {
            "reservationId": reservation["id"],
            "ref": reservation["ref"],
            "status": reservation["status"],
            "checkIn": reservation["check_in"],
            "checkOut": reservation["check_out"],
            "totalPaise": reservation["total_paise"],
            "instruction": (
                "Confirm using the reference. Do not say the booking is paid — "
                "paymentStatus says what has actually been collected."
            ),
        },
        reservation["id"],
    )


@tool(
    name="release_hold",
    description=(
        "Give held rooms back when a guest decides against a quote. Safe to "
        "call — it only ever returns stock."
    ),
    input_model=HoldArgs,
    permission="rates",
    autonomy_floor=2,
    mutating=True,
    usage_type="ai.tool_write",
    result_type="hold",
)
async def release_hold(args: HoldArgs, ctx: AgentContext) -> tuple[dict, str]:
    def _write():
        db = get_db()
        # The property check is here rather than trusted from the argument: a
        # hold id is a uuid, and a uuid from another hotel would otherwise
        # release their stock.
        rows = (
            db.table("inventory_holds")
            .select("id,property_id,status")
            .eq("id", args.hold_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows or rows[0]["property_id"] != ctx.property_id:
            return None
        db.rpc("release_hold", {"p_hold": args.hold_id}).execute()
        return rows[0]

    row = await run_db(_write)
    if row is None:
        raise OutOfScope(
            f"hold {args.hold_id} does not belong to this property",
            "I couldn't find that reservation hold.",
        )
    return {"released": True, "holdId": args.hold_id}, args.hold_id


# ── closing the sale ─────────────────────────────────────────────────────────


def live_quote_id(conversation_id: str | None) -> str | None:
    """
    The active, unexpired quote on this conversation, if there is one.

    **Why a tool argument became optional.** Two production replies were refused
    for *"asked the guest for an internal identifier ('quote ID')"* — the model,
    needing a `quote_id` it could not see, asked the guest to supply one. A guest
    has never seen a quote id and never will; it is a uuid in our database.

    The model was not being obtuse. The argument was required and it did the only
    thing left. `get_booking_status` solved the same problem by taking no
    arguments at all — *"the subject of this tool is the conversation, which the
    model cannot name, cannot guess and cannot be talked into changing"* — and
    this is that idea applied to the quote.

    `status = 'active'` and unexpired, the same window `prompts._live_quote` uses
    to decide what the model is *told* about, so the quote in the prompt and the
    quote this charges cannot be different ones.
    """
    if not conversation_id:
        return None
    from datetime import datetime, timezone

    try:
        rows = (
            get_db().table("quotes").select("id,expires_at")
            .eq("conversation_id", conversation_id)
            .eq("status", "active")
            .order("created_at", desc=True).limit(1)
            .execute().data or []
        )
    except Exception:  # noqa: BLE001
        logger.warning("could not read the live quote for %s", conversation_id,
                       exc_info=True)
        return None
    if not rows:
        return None
    try:
        expires = datetime.fromisoformat(
            str(rows[0]["expires_at"]).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return rows[0]["id"] if expires > datetime.now(timezone.utc) else None


class BookingPaymentLinkArgs(BaseModel):
    quote_id: ResourceId | None = Field(
        default=None,
        description=(
            "Leave this out — the live quote on this conversation is used. Set "
            "it only when the guest is paying for a quote other than the most "
            "recent one, and NEVER ask the guest for it: a quote id is ours, "
            "not something they have ever seen."
        ),
    )
    guest_name: str | None = Field(
        default=None, max_length=120,
        description=(
            "The guest's name, if they have given it in this conversation. "
            "Leave it out otherwise — the name on the thread is used. Never "
            "ask for a name you have already been told."
        ),
    )
    guest_phone: str | None = Field(
        default=None, max_length=32,
        description="Their phone number, if they have given one.",
    )
    guest_email: str | None = Field(
        default=None, max_length=200,
        description="Their email, if they have given one.",
    )


@tool(
    name="create_booking_payment_link",
    description=(
        "Take payment for a quote the guest has agreed to, when they do NOT "
        "already have a booking. Returns a link: the guest taps it, pays by "
        "UPI, card or netbanking, and the booking is confirmed automatically "
        "the moment the payment goes through. Use this for 'book it', 'yes "
        "please', 'how do I pay' on a new stay. Get a quote first. Does NOT "
        "confirm anything by itself — tell the guest their booking is confirmed "
        "once they have paid, never because they say they have."
    ),
    input_model=BookingPaymentLinkArgs,
    # `rates`, matching `create_payment_link`. Asking for money against a price
    # this agent was allowed to quote is part of selling a stay.
    permission="rates",
    # Selling, so the same floor as `get_quote` and `create_booking`. A hotel at
    # level 2 gets an approval request, which is the same answer it gets today.
    autonomy_floor=3,
    mutating=True,
    usage_type="ai.tool_write",
    result_type="payment_link",
    # Deliberately NOT `always_requires_approval`, and this is the one place in
    # the system where that asymmetry is load-bearing, so it is worth stating.
    #
    # `create_booking` carries it because it takes a room off sale on a model's
    # say-so, before any money has moved, and nobody can un-take a night. This
    # tool takes no inventory at all: it raises a Razorpay order, and the
    # reservation is created by `_fulfil_booking` only after a signed webhook
    # says the guest has paid — the identical unattended path the hotel's own
    # booking page has always used. A guest who never pays costs a held quote
    # that expires by itself.
    #
    # Requiring approval here would mean a manager has to press a button before
    # a guest is allowed to *give the hotel money*, which is not a safety
    # property. It is the thing that stopped the AI employee closing a sale.
    always_requires_approval=False,
    requires=(registry.OFF_PLATFORM,),
)
async def create_booking_payment_link(
    args: BookingPaymentLinkArgs, ctx: AgentContext
) -> tuple[dict, str]:
    # The conversation's own quote, when the model did not name one — which is
    # what it should do. See `live_quote_id`.
    quote_id = args.quote_id or await run_db(live_quote_id, ctx.conversation_id)
    if not quote_id:
        raise OutOfScope(
            "no live quote on this conversation to charge for",
            "Let me price those dates for you first — one moment.",
        )

    from app.services.pms import quote_payment

    # The name on the thread, when the model was not given one.
    #
    # It was required, and the model — correctly — asked for it. On a channel
    # that already carries the guest's name that is a round trip for something
    # both sides know: the guest had said "Sahil" four messages earlier, and
    # was asked again at the moment they had agreed to pay. Every extra turn
    # between agreement and payment link is a sale that can go cold.
    name = (args.guest_name or ctx.guest_name or "").strip()
    if not name:
        raise OutOfScope(
            "no name for the booking, on the args or on the conversation",
            "Could you tell me the name the booking should be in?",
        )

    try:
        link = await quote_payment.create_link(
            property_id=ctx.property_id,
            partner_id=ctx.partner_id,
            quote_id=quote_id,
            guest_name=name,
            guest_phone=args.guest_phone or ctx.guest_phone,
            guest_email=args.guest_email,
            source=ctx.channel,
            # Not from the model. The thread is a fact about where this turn is
            # running, and it is what get_booking_status reads back.
            conversation_id=ctx.conversation_id,
        )
    except quote_payment.QuotePaymentError as exc:
        # Already written for a guest to read — see the module. Raised as
        # OutOfScope so the runtime hands the sentence to the model rather than
        # turning it into a failed run the guest sees nothing from.
        raise OutOfScope(f"quote {quote_id}: {exc}", str(exc)) from exc

    # Three amounts, and only one of them is what the link charges.
    #
    # The first version returned `amountText`, `totalText` and `balanceText` as
    # a flat set with an instruction to "give the guest the link and the
    # amount". The model picked the total. A guest was told
    #
    #     Please use this link to complete your booking for ₹7,168
    #
    # and the page it opened asked for ₹1,434, because this property takes a
    # deposit and the rest is paid at the hotel. Both numbers were correct and
    # the sentence was wrong, which is the worst of the three possibilities —
    # a guest who spots it thinks they are being overcharged, and a guest who
    # does not thinks their stay is paid for.
    #
    # So the ready sentence is built here, where the numbers are, and the
    # instruction is to use it rather than to compose one. The fields stay for
    # a question like "what's the total?", which the model should still be able
    # to answer without another tool call.
    # Through `money.display`, not an f-string. Same arithmetic today, and it
    # is the arithmetic that must live in one place — a second copy is a second
    # thing to get wrong the day the format changes.
    amount_text = money.display(link.amount_paise)
    total_text = money.display(link.total_paise)
    due_at_hotel = link.total_paise - link.amount_paise

    if due_at_hotel > 0:
        say = (
            f"{amount_text} to confirm now, and the balance of "
            f"{money.display(due_at_hotel)} at the hotel — total {total_text}."
        )
    else:
        say = f"{amount_text}, which is the full amount."

    return (
        {
            "payUrl": link.pay_url,
            "sayThis": say,
            "payNowText": amount_text,
            "payNowPaise": link.amount_paise,
            "totalText": total_text,
            "dueAtHotelText": money.display(due_at_hotel),
            "instruction": (
                "Send the guest the payUrl and say sayThis, close to word for "
                "word. The link charges payNowText and nothing else — never "
                "tell them to pay totalText through it. Say the booking is "
                "confirmed as soon as the payment goes through. "
                # **There is no reference yet, and saying so stops the model
                # inventing one.**
                #
                # Two production replies were refused for exposing
                # `order_TTNlZOqsCGlaWH` — the Razorpay order id, which the model
                # mined out of the payUrl and offered the guest as a booking
                # reference. It was not being careless: a guest being asked to
                # pay expects a reference, the model had none, and the only
                # identifier in view was the one inside the link.
                #
                # The honest answer is that no reference exists until the money
                # lands: the reservation, and its TB- ref, are created by
                # `_fulfil_booking` when the webhook arrives. Telling the model
                # that is the fix; the guardrail refusing the id afterwards is
                # the backstop, and it stays exactly as it is.
                "There is no booking reference yet — it does not exist until "
                "the payment lands. Do NOT offer the guest a reference, an "
                "order id, or anything out of the link's address. Tell them "
                "they will get their reference the moment the payment goes "
                "through."
            ),
        },
        link.order_id,
    )


# ── did my payment go through? ───────────────────────────────────────────────


class BookingStatusArgs(BaseModel):
    """
    No arguments, and that is the security property.

    The subject of this tool is the conversation, which the model cannot name,
    cannot guess and cannot be talked into changing. "Check booking TB-4K2M9X"
    from a stranger reaches the same code path as "payment hogyi?" from the
    person who made it, and both get this thread's booking or nothing.
    """


def read_booking_state(conversation_id: str, property_id: str) -> dict:
    """
    Compose the money state of one conversation out of the three tables that own
    parts of it.

    Public, and read by the inbox's session panel as well as by the tool below.
    That is deliberate: if staff and the AI composed this state separately they
    would eventually describe the same booking differently, and the guest would
    be holding both descriptions.

    The lifecycle in the brief — DRAFT, QUOTE_CREATED, HOLD_ACTIVE,
    PAYMENT_PENDING, PAYMENT_RECEIVED, CONFIRMED — is not a column anywhere,
    deliberately. `quotes.status`, `payment_orders.status` and
    `reservations.{status,payment_status}` each own a real transition with a
    real writer, and a fourth column restating them would be a fourth opinion
    that drifts. This function is the one place they are read together, and it
    reads them newest-fact-first: a reservation outranks an order, an order
    outranks a quote.
    """
    db = get_db()

    reservation = (
        db.table("reservations")
        .select("id,ref,public_token,status,payment_status,check_in,check_out,"
                "nights,units,adults,children,total_paise,paid_paise,guest_name,"
                "created_at")
        .eq("property_id", property_id)
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute().data or []
    )

    if reservation:
        stay = reservation[0]
        # The folio, not `total - paid`.
        #
        # This computed `max(0, total - paid)` of its own, which is right up
        # until a guest checks in: from then on the extras they order live on
        # the folio and the reservation total does not know about them. The
        # agent would tell an in-house guest they owed ₹4,800 while the desk's
        # screen said ₹6,000, and both would be reading the same booking.
        #
        # One read, one function, four surfaces. See `services/pms/money`.
        folio = (
            db.table("folio_entries").select("kind,amount_paise")
            .eq("reservation_id", stay["id"]).execute().data or []
        )
        purse = money.state(stay, folio)
        total = purse.total_paise
        paid = purse.paid_paise
        state = {
            "confirmed": "CONFIRMED",
            "checked_in": "CHECKED_IN",
            "checked_out": "CHECKED_OUT",
            "cancelled": "CANCELLED",
            "no_show": "CANCELLED",
        }.get(stay.get("status") or "", "CONFIRMED")
        return {
            "state": state,
            "ref": stay.get("ref"),
            "reservationId": stay["id"],
            "guestName": stay.get("guest_name"),
            "checkIn": stay.get("check_in"),
            "checkOut": stay.get("check_out"),
            "nights": stay.get("nights"),
            "rooms": stay.get("units"),
            "adults": stay.get("adults"),
            "children": stay.get("children"),
            "totalPaise": total,
            "paidPaise": paid,
            "refundedPaise": purse.refunded_paise,
            "balancePaise": purse.balance_due_paise,
            "paymentStatus": stay.get("payment_status"),
            # Consumed by `get_booking_status` to build the link and then
            # dropped. It is the guest's own token, but a field the model can
            # quote is a field the model will quote somewhere it does not
            # belong, and the URL already carries everything the guest needs.
            "_publicToken": stay.get("public_token"),
        }

    order = (
        db.table("payment_orders")
        .select("id,status,amount_paise,razorpay_order_id,quote_id,"
                "failure_reason,created_at,paid_at")
        .eq("property_id", property_id)
        .eq("conversation_id", conversation_id)
        .eq("purpose", "booking")
        .order("created_at", desc=True)
        .limit(1)
        .execute().data or []
    )

    if order:
        row = order[0]
        # 'paid' with no reservation on the conversation means fulfilment is
        # mid-flight or failed. Reported as received-not-yet-confirmed, which is
        # exactly what it is — and never as confirmed, because the room has not
        # been taken out of inventory yet.
        state = {
            "created": "PAYMENT_PENDING",
            "paid": "PAYMENT_RECEIVED",
            "failed": "PAYMENT_FAILED",
        }.get(row.get("status") or "", "PAYMENT_PENDING")
        return {
            "state": state,
            "orderId": row.get("razorpay_order_id"),
            "amountPaise": int(row.get("amount_paise") or 0),
            "failureReason": row.get("failure_reason"),
        }

    from datetime import datetime, timezone

    quote = (
        db.table("quotes")
        .select("id,status,check_in,check_out,units,adults,children,"
                "total_paise,expires_at")
        .eq("property_id", property_id)
        .eq("conversation_id", conversation_id)
        .eq("status", "active")
        .order("created_at", desc=True)
        .limit(1)
        .execute().data or []
    )

    if quote:
        row = quote[0]
        try:
            expires = datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00"))
            live = expires > datetime.now(timezone.utc)
        except (ValueError, TypeError, KeyError):
            live = False
        if live:
            return {
                "state": "QUOTE_ACTIVE",
                "quoteId": row["id"],
                "checkIn": row.get("check_in"),
                "checkOut": row.get("check_out"),
                "rooms": row.get("units"),
                "totalPaise": int(row.get("total_paise") or 0),
                "expiresAt": row.get("expires_at"),
            }

    return {"state": "NONE"}


# What the model is told to do about each state, in words it can put in front of
# a guest. Kept out of the system prompt on purpose: a rule about what to say
# when a payment is pending belongs with the tool that discovers it is pending,
# not in a paragraph every turn pays for whether or not money is involved.
_GUIDANCE = {
    "CONFIRMED": (
        "The payment has landed and this booking is real. Tell the guest it is "
        "confirmed, give them the reference, and give them bookingUrl. If "
        "balance is above zero, say `balance` exactly as written is due at "
        "the hotel — never convert a paise figure yourself."
    ),
    "CHECKED_IN": "The guest is currently staying. Answer about their stay, not about booking it.",
    "CHECKED_OUT": "This stay has ended.",
    "CANCELLED": (
        "This booking was cancelled. Do not describe it as active. Anything "
        "about a refund goes to the team — call request_human_handoff."
    ),
    "PAYMENT_RECEIVED": (
        "The money has arrived and the booking is still being written. Tell the "
        "guest the payment came through and their confirmation is moments away. "
        "Do NOT say the booking is confirmed yet, and do NOT ask them to pay again."
    ),
    "PAYMENT_PENDING": (
        "A payment link was sent and nothing has been received against it. Say "
        "the payment has not shown up yet and that you will confirm the moment "
        "it does. Never say it is confirmed because the guest says they paid — "
        "a payment can take a minute, and this tool is what will know."
    ),
    "PAYMENT_FAILED": (
        "The payment did not go through. Say so plainly, without blaming them, "
        "and offer a fresh link."
    ),
    "QUOTE_ACTIVE": (
        "There is a price on the table and no payment. Nothing is booked and no "
        "room is held for them. If they want to go ahead, call "
        "create_booking_payment_link with quoteId."
    ),
    "NONE": (
        "This conversation has no quote, no payment and no booking. If they "
        "believe they have one made elsewhere, ask them for the booking "
        "reference — the short code on their confirmation, like TB-5CECKV — "
        "and hand over to the team. "
        # **Say which words, because the wrong ones get the reply refused.**
        #
        # Observed on a live evaluation run: this guidance said "ask for the
        # booking reference", the model wrote *"kripya reservation ID share
        # karein"*, and the reply guardrail refused it — correctly, because no
        # guest has ever been given a "reservation ID"; that is a column name.
        # The turn then handed a guest to a person over a sentence that was
        # trying to be helpful.
        #
        # The guardrail is right and stays. What was wrong is that one part of
        # the system told the model to ask for something and another refused
        # the words it chose, and the fix belongs where the situation is known.
        "Call it a booking reference or a booking ID and nothing else — never "
        "a 'reservation ID', a 'quote ID' or any other internal name."
    ),
}


@tool(
    name="get_booking_status",
    description=(
        "Where this conversation stands on money: quote, payment and booking. "
        "Call this for ANY question about whether a payment went through, "
        "whether a booking is confirmed, what has been paid, or what is still "
        "owed — including 'payment hogyi', 'have you received it', 'is it "
        "booked'. Never answer those from what the guest tells you; a guest "
        "saying they paid is not the same as the money arriving. Takes no "
        "arguments: it reads this conversation's own booking and nobody else's."
    ),
    input_model=BookingStatusArgs,
    permission=None,
    autonomy_floor=0,
    mutating=False,
    usage_type="ai.tool_read",
    result_type="none",
    scoped_to_conversation=True,
)
async def get_booking_status(args: BookingStatusArgs, ctx: AgentContext) -> dict:
    if not ctx.conversation_id:
        # A console turn with no thread. There is no "this conversation's
        # booking" to report, and inventing one from the property would be the
        # cross-guest leak this tool's argument list exists to prevent.
        raise ToolUnavailable(
            "get_booking_status needs a conversation and this run has none",
        )

    state = await run_db(read_booking_state, ctx.conversation_id, ctx.property_id)

    token = state.pop("_publicToken", None)
    if state.get("ref"):
        from app.services.messaging import service as messaging

        state["bookingUrl"] = messaging.booking_url(
            {"ref": state["ref"], "public_token": token},
        )

    state["instruction"] = _GUIDANCE.get(state["state"], _GUIDANCE["NONE"])
    return state


# ── cancelling ───────────────────────────────────────────────────────────────
#
# The readiness audit found no cancellation anywhere in the system: the AI could
# sell a room in ninety seconds and a guest had to reach a human to undo it.
# Two tools rather than one, and the split is the safety property.
#
#   get_cancellation_quote   what would happen. Reads. Level 0.
#   cancel_booking           make it happen. Writes, releases stock. Level 3.
#
# A guest asking "cancel karne pe kitna wapas milega?" is asking the first
# question and must not be answered by doing the second. Merging them would
# mean the only way to find out the refund was to cancel the booking, which is
# the kind of design that gets a hotel a one-star review.


def _cancellation_reply(payload: dict) -> str:
    """
    What the model is instructed to do with the answer.

    Instructions rather than a script, for the same reason `negotiate_rate` uses
    instructions: the sentence has to be in the guest's language and the agent's
    voice. What is fixed is which *figures* may be said — `refund` and
    `retained`, both formatted here — and that the agent must not soften a
    non-refundable rate into a maybe.
    """
    if payload.get("alreadyCancelled"):
        return (
            "This booking was already cancelled. Say so plainly and give the "
            "reference. Do not cancel anything again and do not promise a "
            "second refund."
        )
    kind = payload.get("kind")
    if kind == "non_refundable":
        return (
            "Cancelled. This rate was non-refundable, so nothing comes back — "
            "say that directly rather than implying it might. Give the "
            "reference. If they push, you may offer to ask the hotel with "
            "escalate_to_team; do not promise an exception yourself."
        )
    if kind == "free":
        return (
            "Cancelled, with a full refund of `refund`. Say that figure and no "
            "other, and tell them the hotel processes it back to the original "
            "payment method — it is not instant."
        )
    return (
        "Cancelled. `refund` comes back and `retained` does not, under the "
        "policy the booking was made on. Say both figures exactly as given, "
        "explain that the deadline had passed if afterDeadline is true, and do "
        "not apologise for the policy or hint that it could be waived."
    )


def _money_out(payload: dict) -> dict:
    """
    Rupees beside every paise figure, so the model never divides.

    The rule from `money.display`: a live replay divided 1920000 by a thousand
    and offered a ₹19,200 stay for ₹1,920. Anything a guest could be told is
    formatted before it reaches the context.
    """
    for paise_key, display_key in (
        ("refundPaise", "refund"), ("retainedPaise", "retained"),
        ("paidPaise", "paid"), ("totalPaise", "total"),
    ):
        if payload.get(paise_key) is not None:
            payload[display_key] = money.display(payload[paise_key])
    return payload


async def _the_conversations_booking(ctx: AgentContext) -> str:
    """
    Which booking this thread is about, from the context and never the model.

    `ctx.reservation_id` when the conversation is pinned to one, otherwise the
    single reservation this guest is allowed to see. Two or more, and the agent
    has to ask which — a guest with a booking in March and one in June saying
    "cancel kar do" has not said enough, and guessing would cancel the wrong
    stay.
    """
    if ctx.reservation_id:
        return ctx.reservation_id

    allowed = await ctx.allowed_reservations()
    live = [r for r in allowed] if allowed else []
    if len(live) == 1:
        return live[0]
    if not live:
        raise OutOfScope(
            "no reservation is attached to this conversation",
            "I can't find a booking on this chat — could you send me the "
            "booking reference?",
        )
    raise OutOfScope(
        f"{len(live)} reservations are in scope and the guest named none",
        "You have more than one booking with us — which one did you want to "
        "cancel? The reference or the dates would help.",
    )


@tool(
    name="get_cancellation_quote",
    description=(
        "What this guest's own booking would get back if they cancelled right "
        "now — the refund, what is retained, and whether the deadline has "
        "passed. Reads only; it cancels nothing. Use this for 'cancel karne pe "
        "paisa wapas milega?' and before any cancellation, so the guest hears "
        "the number before they decide. Never work the refund out yourself."
    ),
    input_model=BookingStatusArgs,
    permission=None,
    autonomy_floor=0,
    mutating=False,
    usage_type="ai.tool_read",
    result_type="none",
    scoped_to_conversation=True,
)
async def get_cancellation_quote(args: BookingStatusArgs, ctx: AgentContext) -> dict:
    from app.services.pms import cancellation as cancellation_service

    reservation_id = await _the_conversations_booking(ctx)
    payload = await cancellation_service.preview(ctx.property_id, reservation_id)
    payload = _money_out(payload)

    if not payload["cancellable"]:
        payload["instruction"] = (
            (payload.get("blockedExplanation") or
             "This booking cannot be cancelled from here.")
            + " Say so and offer escalate_to_team. Do not quote a refund figure."
        )
    else:
        payload["instruction"] = (
            "This is what they would get back — `refund` — and what the hotel "
            "keeps — `retained`. Quote those figures and no others. Nothing has "
            "been cancelled: if they want to go ahead, call cancel_booking."
        )
    return payload


@tool(
    name="cancel_booking",
    description=(
        "Cancel this guest's own booking. Releases the rooms and records what "
        "is refundable under the policy the booking was sold on. Use it only "
        "after the guest has been told the refund and has confirmed they want "
        "to go ahead. It takes no amount and no booking id — both come from the "
        "hotel's records. Never state a refund figure this tool did not return."
    ),
    input_model=CancelArgs,
    # 'frontdesk', the same scope `create_booking` uses. Not a new 'bookings'
    # scope: the permission vocabulary is shared with a human's role, and
    # inventing a scope nobody's role carries would mean no agent could hold it
    # without a migration on `agent_permissions` first.
    permission="frontdesk",
    # Level 3, the same as create_booking and for the mirror-image reason: this
    # gives a room back and gives money back, and a hotel that lets an AI do the
    # second unattended has to have said so.
    autonomy_floor=3,
    mutating=True,
    usage_type="ai.tool_booking",
    result_type="reservation",
    scoped_to_conversation=True,
    when_deferred=(
        "Not cancelled yet — the hotel has to approve a cancellation here. Tell "
        "the guest their request has gone to the team and that nothing has "
        "changed on their booking yet. Do not say it is cancelled and do not "
        "quote a refund."
    ),
    requires=(registry.OFF_PLATFORM,),
)
async def cancel_booking(args: CancelArgs, ctx: AgentContext) -> tuple[dict, str | None]:
    from fastapi import HTTPException

    from app.services.pms import cancellation as cancellation_service

    reservation_id = await _the_conversations_booking(ctx)

    try:
        payload = await cancellation_service.cancel(
            ctx.property_id, reservation_id,
            reason=args.reason,
            actor="agent",
            actor_name=ctx.principal.display_name or "AI agent",
        )
    except HTTPException as exc:
        # A booking that has checked in, checked out or is already a no-show.
        # The guest hears the true reason and gets a person, rather than an
        # apology for something the agent could not name.
        raise OutOfScope(
            f"cancel_booking refused: {exc.detail}",
            "I can't cancel that one from here — let me get somebody from the "
            "hotel to look at it with you.",
        ) from exc

    payload = _money_out(payload)
    payload["currency"] = "INR"
    payload["instruction"] = _cancellation_reply(payload)
    return payload, payload["reservationId"]
