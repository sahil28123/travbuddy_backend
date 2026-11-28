"""
The three things a guest asks for after they have booked.

    "Can you send payment details?"   ──► create_payment_link
    "How do I check in?"              ──► get_check_in_link
    "Can I get airport pickup?"       ──► order_service

Each of them was missing, and their absence had the same shape: the agent could
*talk* about the thing and not do it. It could read the price of an airport
transfer out of `list_services` and had no way to book one; it could confirm a
booking and had no way to collect for it. A guest agent that answers "yes, we
do that" and then hands over to a human is a brochure with a chat box.

**All three are scoped to this conversation's own stay.** None takes a
reservation id the model chose: `ctx.require_reservation()` runs on every one,
so a guest who types a stranger's reference gets the same refusal whatever the
model was persuaded to do with it.

**None of them moves money on trust.** `create_payment_link` raises a Razorpay
order and returns a URL; nothing on this path can mark a stay paid. That only
happens when a signed webhook settles. "I have paid" is a sentence, not a
payment, and there is deliberately no tool that treats it as one.
"""
import logging

from pydantic import BaseModel, Field

from app.db.client import get_db, run_db
from app.services.ai.context import AgentContext
from app.services.ai.errors import OutOfScope, ToolUnavailable
from app.services.ai.ids import ResourceId
from app.services.ai import registry
from app.services.ai.registry import tool
from app.services.pms import money, public_booking, stay_payment

logger = logging.getLogger(__name__)


def _rupees(paise: int) -> str:
    """One spelling of money, shared with every other surface. See money.display."""
    return money.display(paise)


async def _this_stay(ctx: AgentContext, reservation_id: str | None) -> str:
    """
    The stay this conversation is about, whatever the model passed.

    An explicit id is honoured only after `require_reservation` agrees it
    belongs here; with no id at all we fall back to the conversation's own
    link. The fallback is the common case — a guest messaging from the number
    they booked on does not quote their own reference — and having it means the
    model is never tempted to invent one.
    """
    if reservation_id:
        return await ctx.require_reservation(str(reservation_id))
    if ctx.reservation_id:
        return str(ctx.reservation_id)

    allowed = await ctx.allowed_reservations()
    if len(allowed) == 1:
        return next(iter(allowed))
    raise OutOfScope(
        "no reservation is linked to this conversation",
        "I can't see a booking linked to this chat. "
        "Could you tell me the name it was made under?",
    )


# ── paying ───────────────────────────────────────────────────────────────────

class PaymentLinkArgs(BaseModel):
    reservation_id: ResourceId | None = Field(
        default=None,
        description="Leave empty for the booking in this chat.",
    )
    amount_rupees: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Only for a part payment or deposit. Leave empty to ask for the "
            "whole outstanding balance, which is almost always right."
        ),
    )


@tool(
    name="create_payment_link",
    description=(
        "Get a secure payment link for a booking's outstanding balance. Share "
        "the link in your reply — the guest taps it and pays by UPI, card or "
        "netbanking. Use for 'send payment details', 'how do I pay', 'send me "
        "the QR'. Does NOT mark anything paid: say the booking is confirmed "
        "once payment goes through, never because the guest says they have paid."
    ),
    input_model=PaymentLinkArgs,
    # `rates` and not `folio`. Asking for money that is already owed is part of
    # selling a stay; `folio` is the scope that *changes* a bill, and the guest
    # agent is deliberately not trusted with it -- talking an AI into a
    # discount is the attack this separation exists for.
    permission="rates",
    autonomy_floor=2,
    mutating=True,
    usage_type="ai.tool_write",
    result_type="payment_link",
    requires=(registry.OFF_PLATFORM,),
)
async def create_payment_link(args: PaymentLinkArgs, ctx: AgentContext) -> tuple[dict, str]:
    reservation_id = await _this_stay(ctx, args.reservation_id)

    try:
        request = await stay_payment.create_request(
            property_id=ctx.property_id,
            reservation_id=reservation_id,
            amount_paise=args.amount_rupees * 100 if args.amount_rupees else None,
        )
    except stay_payment.PaymentRequestError as exc:
        # Already the sentence to say to a guest -- "that booking is already
        # paid in full" is an answer, not an error.
        raise OutOfScope(f"payment link refused: {exc}", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(
            f"payment link failed: {type(exc).__name__}: {exc}",
            "I couldn't create the payment link just now — "
            "let me get someone from the team to help.",
        ) from exc

    return (
        {
            "payUrl": request.pay_url,
            "amountPaise": request.amount_paise,
            "amountText": _rupees(request.amount_paise),
            "balancePaise": request.balance_paise,
            "ref": request.ref,
            "instruction": (
                "Give the guest the payUrl and the amountText. Tell them it "
                "accepts UPI, card and netbanking, and that the booking is "
                "confirmed once the payment goes through. Do not claim it is "
                "paid yet."
            ),
        },
        # The internal order uuid, not Razorpay's `order_XXXX`.
        # `agent_actions.result_id` is a uuid column.
        request.order_id,
    )


# ── checking in ──────────────────────────────────────────────────────────────

class CheckInLinkArgs(BaseModel):
    reservation_id: ResourceId | None = Field(
        default=None,
        description="Leave empty for the booking in this chat.",
    )


def _read_stay(property_id: str, reservation_id: str) -> dict | None:
    rows = (
        get_db().table("reservations")
        .select("id,ref,status,check_in,check_out,guest_name")
        .eq("id", reservation_id).eq("property_id", property_id)
        .limit(1).execute().data or []
    )
    return rows[0] if rows else None


@tool(
    name="get_check_in_link",
    description=(
        "Get the online check-in link for a booking. The guest fills in their "
        "details before arrival so the desk is quick. Use for 'how do I check "
        "in', 'can I check in online', 'web check-in'. Share the link in your "
        "reply."
    ),
    input_model=CheckInLinkArgs,
    permission="frontdesk",
    # Reading a link that the guest is entitled to. Nothing moves.
    autonomy_floor=1,
    mutating=False,
    usage_type="ai.tool_read",
    result_type="check_in_link",
    requires=(registry.OFF_PLATFORM,),
)
async def get_check_in_link(args: CheckInLinkArgs, ctx: AgentContext) -> tuple[dict, str]:
    from app.config import settings

    reservation_id = await _this_stay(ctx, args.reservation_id)
    stay = await run_db(_read_stay, ctx.property_id, reservation_id)
    if not stay:
        raise OutOfScope(
            f"reservation {reservation_id} not found",
            "I can't find that booking — let me get someone to check.",
        )
    if stay["status"] in ("cancelled", "no_show"):
        raise OutOfScope(
            f"reservation {reservation_id} is {stay['status']}",
            "That booking has been cancelled, so there's nothing to check in to.",
        )
    if stay["status"] in ("checked_out",):
        raise OutOfScope(
            "stay already checked out",
            "That stay has already finished — hope to see you again soon!",
        )

    base = (settings.pms_app_url or "").rstrip("/")
    return (
        {
            "checkInUrl": f"{base}/checkin/{stay['ref']}",
            "ref": stay["ref"],
            "checkIn": stay["check_in"],
            "instruction": (
                "Share the checkInUrl. Tell them it takes a minute and means "
                "they can go straight to their room on arrival."
            ),
        },
        stay["ref"],
    )


# ── buying something during the stay ─────────────────────────────────────────

class OrderServiceArgs(BaseModel):
    service_id: ResourceId = Field(description="From list_services. Never invent one.")
    quantity: int = Field(default=1, ge=1, le=20)
    service_date: str | None = Field(
        default=None,
        description="YYYY-MM-DD. Leave empty for the day the guest arrives.",
    )
    reservation_id: ResourceId | None = Field(
        default=None,
        description="Leave empty for the booking in this chat.",
    )


@tool(
    name="order_service",
    description=(
        "Book one of the hotel's services onto a guest's stay — airport "
        "pickup, spa, dinner, late checkout, a tour. Use ONLY a service_id "
        "from list_services, and tell the guest the price before you book it. "
        "The charge goes on their room bill."
    ),
    input_model=OrderServiceArgs,
    permission="rates",
    # Level 3: it puts a charge on somebody's bill. At the guest agent's
    # shipping level of 2 this produces an approval request, which is the
    # right default for a hotel that has not yet decided to let an AI spend
    # its guests' money unwatched.
    autonomy_floor=3,
    mutating=True,
    usage_type="ai.tool_write",
    result_type="service_order",
)
async def order_service(args: OrderServiceArgs, ctx: AgentContext) -> tuple[dict, str]:
    from datetime import date

    reservation_id = await _this_stay(ctx, args.reservation_id)
    stay = await run_db(_read_stay, ctx.property_id, reservation_id)
    if not stay:
        raise OutOfScope(
            f"reservation {reservation_id} not found",
            "I can't find that booking — let me get someone to check.",
        )
    if stay["status"] in ("cancelled", "no_show", "checked_out"):
        raise OutOfScope(
            f"cannot add a service to a {stay['status']} stay",
            "I can't add that to this booking — let me get someone from the team.",
        )

    check_in = date.fromisoformat(str(stay["check_in"]))
    check_out = date.fromisoformat(str(stay["check_out"]))
    when = date.fromisoformat(args.service_date) if args.service_date else check_in
    # Half-open, matching `price_addons`. A guest is not here on the morning
    # they leave, so a transfer booked for the checkout date is a job nobody
    # turns up for. Checked here as well as there so the guest gets this
    # sentence rather than the booking engine's 400.
    if not (check_in <= when < check_out):
        raise OutOfScope(
            f"{when} is outside the stay {check_in}..{check_out}",
            f"That date isn't during your stay — you're with us from "
            f"{check_in:%d %b} to {check_out:%d %b}.",
        )

    # Priced by the booking engine, not here. Same function the website's
    # add-on picker uses, so a spa booked in chat and a spa booked on the site
    # cost the same and consume the same daily capacity.
    try:
        priced = await public_booking.price_addons(
            ctx.property_id,
            [{
                "serviceId": args.service_id,
                "quantity": args.quantity,
                "serviceDate": when.isoformat(),
            }],
            check_in=check_in,
            check_out=check_out,
            timezone_name="Asia/Kolkata",
        )
    except Exception as exc:  # noqa: BLE001
        raise OutOfScope(
            f"service {args.service_id} could not be priced: {exc}",
            "I couldn't book that just now — let me get someone from the team "
            "to sort it out for you.",
        ) from exc

    if not priced:
        raise OutOfScope(
            f"service {args.service_id} is not available",
            "That isn't something we can add just now — "
            "let me check with the team.",
        )

    reservation = await run_db(
        lambda: get_db().table("reservations").select("*")
        .eq("id", reservation_id).limit(1).execute().data[0]
    )
    written = await public_booking.attach_addons(
        reservation, [p.as_dict() for p in priced]
    )

    total = sum(p.total_paise for p in priced)
    line = priced[0]
    return (
        {
            "name": line.name,
            "quantity": line.quantity,
            "date": when.isoformat(),
            "totalPaise": total,
            "totalText": _rupees(total),
            "instruction": (
                "Confirm what you booked, for when, and that the charge is on "
                "their room bill to settle at checkout."
            ),
        },
        (written[0]["id"] if written else reservation_id),
    )


# ── carrying a conversation onto the website ─────────────────────────────────

class HotelPageArgs(BaseModel):
    """
    A room to open on, and nothing else.

    **There is no field for a date, a party size or a URL**, and that is the
    whole design. The dates go on the link because the *server* reads them off
    this conversation's session — which was written from tool results, not from
    what the model believed — so a link cannot carry dates the guest never gave.
    """
    room_type_id: ResourceId | None = Field(
        default=None,
        description=(
            "Optional. Open the page for one room type instead of the hotel's "
            "main page. Use the id from get_room_types, never a room name."
        ),
    )


# The booking parameters `/book/[slug]` reads. Kept as a closed list here for
# the same reason the page keeps one: an open query string on an indexable
# template is an unbounded set of duplicate URLs.
_PAGE_PARAMS = (
    ("check_in", "checkin"),
    ("check_out", "checkout"),
    ("adults", "adults"),
    ("children", "children"),
    ("units", "rooms"),
)


def _hotel_page(property_id: str, conversation_id: str | None,
                room_type_id: str | None) -> dict:
    from urllib.parse import urlencode

    from app.config import settings

    db = get_db()
    prop = (
        db.table("properties").select("slug,name,status")
        .eq("id", property_id).limit(1).execute().data or [{}]
    )[0]
    site = (
        db.table("hotel_websites").select("status,published")
        .eq("property_id", property_id).limit(1).execute().data or []
    )
    live = (
        prop.get("status") == "active"
        and prop.get("slug")
        and site and site[0].get("status") == "published" and site[0].get("published")
    )
    if not live:
        # The same rule `resolve_published` enforces, applied one step earlier.
        # A hotelier drafting a site has not agreed to anybody seeing it, and a
        # link to a 404 is worse than saying there is no page yet.
        return {"url": "", "instruction": (
            "This hotel has no published page yet. Do NOT invent a website "
            "address or send one. Answer the guest here instead."
        )}

    path = f"/book/{prop['slug']}"
    if room_type_id:
        room = (
            db.table("room_types").select("slug,code,name,is_active")
            .eq("id", room_type_id).eq("property_id", property_id)
            .limit(1).execute().data or []
        )
        # Another hotel's room id, or one that is off sale, silently falls back
        # to the hotel's own page rather than producing a link to nothing.
        if room and room[0].get("is_active"):
            leaf = room[0].get("slug") or (room[0].get("code") or "").lower()
            if leaf:
                path = f"{path}/rooms/{leaf}"

    query: dict[str, str] = {}
    if conversation_id:
        rows = (
            db.table("conversations").select("guest_profile")
            .eq("id", conversation_id).eq("property_id", property_id)
            .limit(1).execute().data or []
        )
        profile = (rows[0].get("guest_profile") if rows else None) or {}
        if isinstance(profile, dict):
            for stored, param in _PAGE_PARAMS:
                value = profile.get(stored)
                if value not in (None, "", 0):
                    query[param] = str(value)

    url = f"{settings.site_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    return {
        "url": url,
        "prefilled": sorted(query),
        "instruction": (
            "Share this url as it is written. It opens this hotel's own page "
            + ("with the guest's dates and party already filled in, so they do "
               "not retype anything. " if query else "")
            + "Do not edit the link and do not add anything to it."
        ),
    }


@tool(
    name="get_hotel_page_link",
    description=(
        "Get a link to this hotel's own web page, with the dates and party "
        "this guest has already given filled in. Use when a guest wants to "
        "see photos, rooms, facilities or prices laid out, or says they will "
        "book later. Takes no dates: it reads them from this conversation."
    ),
    input_model=HotelPageArgs,
    permission=None,
    # A link to a public page. Nothing moves and nothing is spent.
    autonomy_floor=0,
    mutating=False,
    usage_type="ai.tool_read",
    result_type="none",
    scoped_to_conversation=True,
    requires=(registry.OFF_PLATFORM,),
)
async def get_hotel_page_link(args: HotelPageArgs, ctx: AgentContext) -> dict:
    """
    Carry the conversation onto the website instead of restarting it there.

    **The failure this closes is retyping.** A guest who has spent six messages
    settling on 11–15 September for four people, and is then sent a bare hotel
    URL, arrives at an empty date picker — and the most common thing to do with
    an empty date picker is close the tab.

    **The model names a room; it never names a URL.** Same property as
    `send_room_photos`, and for the same reason: every part of this link is
    composed here, from this property's slug and this conversation's session.
    There is no argument through which a model — or a guest talking to one —
    can put a different address in front of somebody.
    """
    return await run_db(
        _hotel_page, ctx.property_id, ctx.conversation_id,
        str(args.room_type_id) if args.room_type_id else None,
    )
