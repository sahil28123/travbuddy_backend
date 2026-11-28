"""
The messaging service. Channel-agnostic, and the only thing that decides
whether a message may be sent.

Four responsibilities, and they are here rather than spread out for a reason:

**Property identity.** `resolve_connection()` turns a provider account id into
exactly one property, through `resolve_whatsapp_property()`. No caller anywhere
accepts a property from a request body.

**Idempotency, inbound.** `record_inbound()` is keyed on `(channel,
external_id)`. Meta redelivers; the second delivery returns the row it already
wrote, flagged `duplicate`, and the pipeline stops there. Without it one guest
message becomes two agent runs, two tickets and two replies.

**Idempotency, outbound.** `send()` writes a `queued` row with a
`client_reference` *before* the provider call, then updates it. A retry after a
timeout finds that row and does not send again.

**The window.** WhatsApp permits free-form replies only for 24 hours after the
guest's last message. `send()` refuses outside it with `MESSAGE_WINDOW_CLOSED`,
which the agent receives as a structured code it can act on. The model is never
asked whether the window is open, because a model asked that question would
sometimes say yes.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.db.client import get_db, run_db
from app.services import billing
from app.services.messaging import credentials as creds_module
from app.services.messaging import language as language_module
from app.services.messaging import policy
from app.services.messaging import session as session_module
from app.services.messaging import templates as templates_module
from app.services.messaging.channels import (
    INSUFFICIENT_CREDITS,
    NO_CONNECTION,
    PROVIDER_FAILED,
    TEMPLATE_NOT_FOUND,
    WINDOW_CLOSED,
    DeliveryUpdate,
    InboundMessage,
    MessagingError,
)

logger = logging.getLogger(__name__)


@dataclass
class RecordedInbound:
    conversation: dict
    message: dict
    duplicate: bool
    # True when this message opened a *new* service window rather than arriving
    # inside one already open. On WhatsApp nothing needs it — Meta tells us on
    # the status callback and bills the conversation itself. On Instagram there
    # is no such callback and no Meta conversation charge, so this is the only
    # honest moment at which to open one of ours.
    opened_window: bool = False


@dataclass
class SentMessage:
    message: dict
    external_id: str | None
    template_name: str | None
    duplicate: bool = False


# ── connections ──────────────────────────────────────────────────────────────

def _resolve_connection(account_id: str, channel: str) -> dict | None:
    pol = policy.for_channel(channel)
    if pol.resolver_rpc is None:
        return _resolve_from_table(pol, account_id)
    rows = get_db().rpc(pol.resolver_rpc, {pol.resolver_arg: account_id}).execute().data or []
    if not rows:
        return None
    row = dict(rows[0])
    # The RPC returns the connection's own id under a different name than the
    # table does. Normalised here so callers see one shape.
    row["id"] = row.pop("connection_id")
    row[pol.account_column] = account_id
    row["channel"] = channel
    return row


def _resolve_from_table(pol: policy.ChannelPolicy, account_id: str) -> dict | None:
    """
    The OTA resolver: `repull_connections.workspace_id` → the hotel.

    A table read rather than a definer function, for the reason `policy.py`
    gives on `resolver_rpc`: the row is server-written and the id is Repull's,
    delivered inside a signed webhook. Returned in the same shape the RPCs
    produce, so `record_inbound` cannot tell the two routes apart.
    """
    db = get_db()
    rows = (
        db.table(pol.connections_table).select("*")
        .eq(pol.account_column, account_id)
        .in_("status", list(pol.live_statuses))
        .limit(1).execute().data or []
    )
    if not rows:
        return None
    row = dict(rows[0])
    row["id"] = row.get(pol.connection_key)
    if not row.get("partner_id"):
        # `repull_connections.partner_id` is nullable — a workspace can be
        # created before the property row has been read — and a conversation
        # cannot be. The property is the authority on whose hotel it is.
        prop = (
            db.table("properties").select("partner_id")
            .eq("id", row["property_id"]).limit(1).execute().data or []
        )
        if not prop:
            return None
        row["partner_id"] = prop[0]["partner_id"]
    row["channel"] = pol.channel
    return row


async def resolve_connection(account_id: str, channel: str = policy.WHATSAPP) -> dict:
    """
    Which property owns this provider account. The only trusted answer.

    Raises when the account is unknown, which is the correct response to a
    webhook for an account nobody has connected: we cannot say whose guest that
    is, and guessing would put a stranger's message into a hotel's inbox.
    """
    row = await run_db(_resolve_connection, account_id, channel)
    if not row:
        raise MessagingError(
            NO_CONNECTION,
            f"no property is connected to {channel} account {account_id}",
        )
    return row


def _connection_for_property(property_id: str, channel: str = policy.WHATSAPP) -> dict | None:
    pol = policy.for_channel(channel)
    rows = (
        get_db()
        .table(pol.connections_table)
        .select("*")
        .eq("property_id", property_id)
        .in_("status", list(pol.live_statuses))
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    row = dict(rows[0])
    row.setdefault("id", row.get(pol.connection_key))
    # Carried so the adapter factory does not have to guess which provider a
    # connection row came from — the two tables have different id columns and a
    # row that has lost its channel is a row that can be sent to the wrong API.
    row["channel"] = channel
    return row


# ── conversations ────────────────────────────────────────────────────────────

def _upsert_conversation(
    *, property_id: str, partner_id: str, channel: str,
    external_id: str, sender_external_id: str, sender_name: str | None,
    account_id: str,
) -> dict:
    db = get_db()

    existing = (
        db.table("conversations")
        .select("*")
        .eq("property_id", property_id)
        .eq("channel", channel)
        .eq("external_id", external_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if existing:
        return existing[0]

    # Adopt a thread we started ourselves.
    #
    # `conversations_identity_idx` is keyed on `external_id`, which only exists
    # once the guest has written — so a conversation opened by an outbound
    # campaign has none, and this guest's first reply would have created a
    # *second* row. That is not cosmetic: the reply would be "STOP", and it
    # would land on a thread with no history while the campaign thread sat
    # beside it looking unanswered.
    #
    # Matched on the address rather than on the guest record, because the
    # address is what both rows are certain to share.
    if sender_external_id:
        address_column = "guest_phone" if channel == policy.WHATSAPP else "guest_handle"
        orphan = (
            db.table("conversations")
            .select("*")
            .eq("property_id", property_id)
            .eq("channel", channel)
            .eq(address_column, sender_external_id)
            .is_("external_id", "null")
            .order("created_at")
            .limit(1)
            .execute()
            .data
            or []
        )
        if orphan:
            adopted = (
                db.table("conversations")
                .update({"external_id": external_id, "channel_account_id": account_id})
                .eq("id", orphan[0]["id"])
                .execute()
                .data
            )
            return adopted[0] if adopted else orphan[0]

    # A returning guest, matched on whatever this channel's sender id actually
    # is. Scoped to the property — the same person at two hotels is two guest
    # records, and joining them would be a cross-property leak dressed up as a
    # convenience.
    #
    # **The column differs by channel and that is not cosmetic.** On WhatsApp
    # the sender id is a phone number, so the phone column is the right one. On
    # Instagram it is a 17-digit IGSID, which matched no phone number that has
    # ever existed — so every returning Instagram guest arrived as a stranger,
    # `guest_id` stayed null, `_link_reservation` never ran, and a guest who had
    # booked through this same thread was told nothing was found.
    guest_id = None
    # Not on an OTA thread. The sender id there is a Repull conversation id —
    # six or seven digits that would sail through the phone match below and
    # hand this thread to whichever guest's number happens to end in them. The
    # guest on an OTA thread is found through the booking instead: see
    # `_link_ota_reservation`, which reads Repull's own reservation id.
    if sender_external_id and channel != policy.OTA:
        found = []
        if channel == "instagram":
            found = (
                db.table("guests")
                .select("id,name")
                .eq("property_id", property_id)
                .eq("instagram_id", sender_external_id)
                .limit(1)
                .execute()
                .data
                or []
            )
        else:
            # Digits only, not `lstrip("+")`.
            #
            # This value goes into an `ilike` pattern, where `%` and `_` are
            # wildcards — and a sender id of `%%%%%%%%%%` would then match the
            # first guest at this property, handing one guest another's name and
            # booking history. It arrives inside a signature-verified Meta
            # payload today, so this is not a live hole; it is one webhook
            # change away from being one, and the fix costs a filter. The same
            # class of bug already cost this codebase a coupon bypass.
            digits = "".join(c for c in sender_external_id if c.isdigit())
            found = (
                db.table("guests")
                .select("id,name")
                .eq("property_id", property_id)
                .ilike("phone", f"%{digits[-10:]}")
                .limit(1)
                .execute()
                .data
                or []
            ) if len(digits) >= 8 else []
        if found:
            guest_id = found[0]["id"]
            sender_name = sender_name or found[0].get("name")

    row = {
        "property_id": property_id,
        "partner_id": partner_id,
        "channel": channel,
        "external_id": external_id,
        "channel_account_id": account_id,
        "guest_phone": sender_external_id if channel == "whatsapp" else None,
        # On an OTA thread the "handle" is the Repull conversation id: it is
        # what `send()` passes as `to`, and the only address the guest has —
        # the OTA keeps their real one to itself.
        "guest_handle": sender_external_id if channel in ("instagram", policy.OTA) else None,
        "guest_name": sender_name,
        "guest_id": guest_id,
        "handler": "ai",
        "status": "open",
    }
    return db.table("conversations").insert(row).execute().data[0]


def _conversation_for_guest(property_id: str, channel: str, guest: dict) -> dict | None:
    """
    The thread to write to when *we* are starting the conversation.

    Every conversation this codebase has created until now began with an inbound
    message, so `_upsert_conversation` keys on the provider's thread id. A
    campaign has no such id — nobody has written — so this is the other door,
    and it is deliberately narrow: it takes a guest row that a segment query
    already returned, never an address from a caller.

    **Instagram cannot use it to start anything.** There is no way to open an
    Instagram thread from a handle; Meta only lets you reply into one the person
    opened. So on Instagram this returns an existing conversation or None, and
    the recipient is skipped as NO_CONTACT. Returning a fresh row there would be
    manufacturing a thread that can never be delivered into.
    """
    db = get_db()
    address = guest.get("phone") if channel == policy.WHATSAPP else None
    address_column = "guest_phone" if channel == policy.WHATSAPP else "guest_handle"

    query = (
        db.table("conversations").select("*")
        .eq("property_id", property_id).eq("channel", channel)
        .eq("guest_id", guest["id"])
        .order("last_message_at", desc=True).limit(1)
    )
    found = query.execute().data or []
    if found:
        return found[0]

    if address:
        by_address = (
            db.table("conversations").select("*")
            .eq("property_id", property_id).eq("channel", channel)
            .eq(address_column, address).limit(1).execute().data or []
        )
        if by_address:
            return by_address[0]

    if not address:
        return None

    prop = (
        db.table("properties").select("partner_id")
        .eq("id", property_id).limit(1).execute().data or []
    )
    if not prop:
        return None

    return db.table("conversations").insert({
        "property_id": property_id,
        "partner_id": prop[0]["partner_id"],
        "channel": channel,
        # No external_id: the provider has not given us a thread id because
        # there is no thread yet. `_upsert_conversation` adopts this row when
        # the guest replies — see the comment there.
        "external_id": None,
        "guest_phone": address if channel == policy.WHATSAPP else None,
        "guest_handle": address if channel != policy.WHATSAPP else None,
        "guest_name": guest.get("name"),
        "guest_id": guest["id"],
        "handler": "ai",
        "status": "open",
    }).execute().data[0]


async def conversation_for_guest(property_id: str, channel: str, guest: dict) -> dict | None:
    """See `_conversation_for_guest`. None means this guest is unreachable here."""
    return await run_db(_conversation_for_guest, property_id, channel, guest)


def _link_reservation(conversation: dict) -> dict:
    """
    Attach the guest's current stay, if they have one.

    This is what makes "my AC isn't working" land on the right door and what
    scopes `get_reservation` in the agent context. Only ever the guest's *own*
    stay: the lookup starts from `conversation.guest_id`, so there is no path
    from a phone number to somebody else's booking.
    """
    if conversation.get("reservation_id") or not conversation.get("guest_id"):
        return conversation

    db = get_db()
    from app.services.pms import clock

    today = clock.business_date_iso(conversation["property_id"])
    rows = (
        db.table("reservations")
        .select("id,status,check_in,check_out")
        .eq("property_id", conversation["property_id"])
        .eq("guest_id", conversation["guest_id"])
        .in_("status", ["confirmed", "checked_in"])
        .gte("check_out", today)
        .order("check_in")
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        return conversation

    updated = (
        db.table("conversations")
        .update({"reservation_id": rows[0]["id"]})
        .eq("id", conversation["id"])
        .execute()
        .data
    )
    return updated[0] if updated else conversation


def _link_ota_reservation(conversation: dict, reservation_id) -> dict:
    """
    Attach the OTA booking this thread belongs to, by Repull's own id.

    `_link_reservation` starts from `guest_id`, which an OTA thread does not
    have: the OTA hides the guest's phone and email, so there is nothing to
    match a guest on. What it does have is better — Repull says which
    reservation the conversation is about, and the booking import stored that
    id as `reservations.external_id`. So the stay, and through it the guest,
    come from the booking rather than from a guess.

    A no-op when the message arrived before its booking was imported (Repull
    sends `reservationId: null` for an inquiry). The next message on the thread
    links it, which is soon enough: nothing the agent needs from the stay is
    needed before the guest has one.
    """
    if conversation.get("reservation_id") or reservation_id in (None, ""):
        return conversation

    db = get_db()
    rows = (
        db.table("reservations").select("id,guest_id")
        .eq("property_id", conversation["property_id"])
        .eq("external_id", str(reservation_id))
        .limit(1).execute().data or []
    )
    if not rows:
        return conversation

    patch = {"reservation_id": rows[0]["id"]}
    if rows[0].get("guest_id") and not conversation.get("guest_id"):
        patch["guest_id"] = rows[0]["guest_id"]
    updated = (
        db.table("conversations").update(patch)
        .eq("id", conversation["id"]).execute().data
    )
    return updated[0] if updated else {**conversation, **patch}


# ── inbound ──────────────────────────────────────────────────────────────────

def _record_inbound(inbound: InboundMessage, connection: dict) -> RecordedInbound:
    db = get_db()
    pol = policy.for_channel(inbound.channel)

    # Idempotency first, before anything is created. A redelivered webhook must
    # not even create a conversation, because creating one would move
    # `last_inbound_at` and reopen a window that had legitimately closed.
    seen = (
        db.table("conversation_messages")
        .select("*")
        .eq("channel", inbound.channel)
        .eq("external_id", inbound.external_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if seen:
        conversation = (
            db.table("conversations").select("*")
            .eq("id", seen[0]["conversation_id"]).limit(1).execute().data[0]
        )
        # A redelivery never reopens a window and never opens a charge.
        return RecordedInbound(conversation=conversation, message=seen[0], duplicate=True)

    conversation = _upsert_conversation(
        property_id=connection["property_id"],
        partner_id=connection["partner_id"],
        channel=inbound.channel,
        external_id=inbound.sender_external_id,
        sender_external_id=inbound.sender_external_id,
        sender_name=inbound.sender_name,
        account_id=connection.get(pol.account_column) or "",
    )
    if inbound.channel == policy.OTA:
        conversation = _link_ota_reservation(conversation, inbound.raw.get("reservationId"))
    conversation = _link_reservation(conversation)

    # Was a window already open when this arrived? Computed before the update
    # below overwrites `last_inbound_at`, which is the only moment the answer
    # exists.
    opened_window = not window_open(conversation)

    sent_at = (inbound.sent_at or datetime.now(timezone.utc)).isoformat()
    try:
        message = db.table("conversation_messages").insert({
            "conversation_id": conversation["id"],
            "property_id": conversation["property_id"],
            "partner_id": conversation["partner_id"],
            "channel": inbound.channel,
            "direction": "inbound",
            "author": "guest",
            "author_name": inbound.sender_name or conversation.get("guest_name"),
            "sender_external_id": inbound.sender_external_id,
            "message_type": inbound.message_type,
            "body": inbound.body,
            "media_urls": list(inbound.media_urls),
            "external_id": inbound.external_id,
            # Inbound messages have no delivery lifecycle of their own: it is here,
            # which is as delivered as it gets.
            "delivery_status": "delivered",
            "metadata": inbound.raw,
            "sent_at": sent_at,
        }).execute().data[0]
    except Exception as exc:  # noqa: BLE001
        # Lost the race, which is not a failure.
        #
        # The select at the top of this function closes the window for
        # redeliveries arriving seconds apart, which is almost all of them. It
        # cannot close the window for two arriving in the same millisecond: both
        # pass the select, both insert, and `conversation_messages_provider_idx`
        # — UNIQUE on (channel, external_id) — refuses the second.
        #
        # That was a 500 back to Meta. Meta redelivers, the redelivery reads as
        # a duplicate, and the guest is answered — so the design worked and left
        # an exception in the log for something it handles. Worse, the 500 comes
        # back for the *whole batch*, so one racing message could cost the other
        # five in the same request their acknowledgement.
        #
        # The index is the real idempotency here and always was. This reads its
        # answer rather than crashing on it.
        if "duplicate key" not in str(exc) and "23505" not in str(exc):
            raise
        seen = (
            db.table("conversation_messages").select("*")
            .eq("channel", inbound.channel)
            .eq("external_id", inbound.external_id)
            .limit(1).execute().data or []
        )
        if not seen:
            raise
        logger.info("inbound %s %s was recorded by a concurrent delivery",
                    inbound.channel, inbound.external_id)
        return RecordedInbound(conversation=conversation, message=seen[0], duplicate=True)

    # The window opens here. Everything about whether we may reply freely for
    # the next 24 hours is this column.
    db.table("conversations").update({
        "last_inbound_at": sent_at,
        "last_message_at": sent_at,
        "last_message": (inbound.body or f"[{inbound.message_type}]")[:200],
        "unread_count": (conversation.get("unread_count") or 0) + 1,
    }).eq("id", conversation["id"]).execute()

    db.table(pol.connections_table).update(
        {"last_inbound_at": sent_at}
    ).eq(pol.connection_key, connection["id"]).execute()

    conversation = (
        db.table("conversations").select("*")
        .eq("id", conversation["id"]).limit(1).execute().data[0]
    )
    return RecordedInbound(conversation=conversation, message=message, duplicate=False,
                           opened_window=opened_window)


async def record_inbound(inbound: InboundMessage, connection: dict) -> RecordedInbound:
    recorded = await run_db(_record_inbound, inbound, connection)

    # Opting out happens here, and it is the earliest honest place.
    #
    # This is the one function every inbound message on every channel passes
    # through — the mirror of `send()` on the way out — so a stop request is
    # acted on before the pipeline is dispatched, before an agent run is opened,
    # before a model reads a word of it, and regardless of whether a human has
    # taken the conversation over or the AI is switched off for this thread.
    #
    # Putting it in a tool would have made the model the thing that noticed,
    # which is exactly the arrangement §14 of the brief forbids: an opt-out that
    # depends on a model behaving is not an opt-out.
    #
    # A duplicate delivery re-applies it, which is harmless — `opt_out` is a
    # write of the same three values — and is better than the alternative of a
    # redelivered STOP being ignored because the first copy was lost.
    if not recorded.duplicate or inbound.body:
        try:
            from app.services.marketing import consent as consent_module

            await consent_module.apply_inbound(recorded.conversation, inbound.body)
        except Exception:  # noqa: BLE001
            # Logged loudly and swallowed. A consent write that failed must not
            # discard the guest's message — but it is the one failure here that
            # somebody has to see, because the guest believes they have stopped.
            logger.exception(
                "could not apply an opt-out from message %s on conversation %s",
                recorded.message.get("id"), recorded.conversation.get("id"),
            )

    # Creator attribution across a channel change.
    #
    # A guest who followed a creator's link is given a WhatsApp button whose
    # prefilled text carries `CR-7X2K9M` — a token this server minted. When the
    # message arrives we read our own token back out of it and bind the thread
    # to the chain, which is what makes PRD §16's common path work: click a
    # reel, message the hotel a fortnight later, book, and the creator is still
    # paid. The guest is asked for nothing and no number is stored early.
    #
    # Every inbound message on every channel passes through here, and almost
    # every one of them costs a single regex that matches nothing. Failures are
    # swallowed for the same reason the consent write's are not: an opt-out that
    # was lost is a guest still being messaged, while a commission that was lost
    # is a row a hotel can fix — and neither is worth discarding the message
    # itself over.
    if not recorded.duplicate and inbound.body:
        try:
            from app.services.pms import creators as creators_service

            await creators_service.bind_conversation(
                recorded.conversation["property_id"],
                recorded.conversation["id"],
                inbound.body,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "could not bind creator attribution on conversation %s",
                recorded.conversation.get("id"), exc_info=True,
            )

    return recorded


# ── the window ───────────────────────────────────────────────────────────────

def window_open(conversation: dict, *, now: datetime | None = None) -> bool:
    """
    Whether free-form outbound is permitted.

    Meta's rule, not ours: 24 hours from the guest's last inbound message. A
    conversation the guest has never written to has no window at all — which is
    correct, and is why a hotel cannot use this to message people who did not
    contact them.
    """
    # An OTA thread never closes: the host may answer the guest at any time.
    # Asked before the closing time, which is None here and would read as shut.
    if policy.always_open(conversation.get("channel")):
        return True
    closes = window_closes_at(conversation)
    if closes is None:
        return False
    return (now or datetime.now(timezone.utc)) < closes


def window_closes_at(conversation: dict) -> datetime | None:
    """
    When free-form outbound stops being permitted.

    The number of hours comes from the channel policy, not from here. WhatsApp
    answers with a constant; Instagram answers 24 hours for the agent and seven
    days once a person has taken over, which is Meta's human-agent allowance and
    is keyed on the server's own `handler` column rather than on anything the
    model can influence.
    """
    channel = conversation.get("channel")
    # A channel with no window has no closing time, and that is an answer rather
    # than an error. The staff console is the case: no provider, no Meta, no
    # 24-hour rule. `policy.for_channel` still raises for a channel nobody has
    # configured, which is the distinction worth keeping.
    if not policy.has_window(channel):
        return None

    last = conversation.get("last_inbound_at")
    if not last:
        return None
    if isinstance(last, str):
        last = datetime.fromisoformat(last.replace("Z", "+00:00"))
    hours = policy.for_channel(channel).window_hours(conversation)
    return last + timedelta(hours=hours)


# ── outbound ─────────────────────────────────────────────────────────────────

def _queue_outbound(conversation: dict, spec: dict) -> tuple[dict, bool]:
    """
    `spec` is a dict rather than keyword arguments because `run_db` forwards
    positionally only. Worth the small awkwardness: the alternative is a
    partial() at every call site or a second wrapper nobody would keep in sync.
    """
    body = spec["body"]
    template = spec["template"]
    image_url = spec.get("image_url")
    client_reference = spec["client_reference"]
    db = get_db()

    # The retry path. A `client_reference` we have already written means this
    # send has been attempted; whether it reached Meta is what the row says.
    seen = (
        db.table("conversation_messages")
        .select("*")
        .eq("property_id", conversation["property_id"])
        .eq("client_reference", client_reference)
        .limit(1)
        .execute()
        .data
        or []
    )
    if seen:
        return seen[0], True

    row = db.table("conversation_messages").insert({
        "conversation_id": conversation["id"],
        "property_id": conversation["property_id"],
        "partner_id": conversation["partner_id"],
        "channel": conversation["channel"],
        "direction": "outbound",
        "author": spec["author"],
        "author_name": spec["author_name"],
        "author_member": spec["author_member"],
        "message_type": "image" if image_url else "text",
        "body": body,
        # The picture itself, on the column the inbound path has always used for
        # one. A photograph a hotel sent is evidence in the same way a
        # photograph a guest sent is.
        #
        # Empty list, not null: the column is NOT NULL with a `{}` default, and
        # writing an explicit null overrides the default rather than falling
        # back to it — which is exactly how a text message started failing to
        # persist the moment this key was added.
        "media_urls": [image_url] if image_url else [],
        "template_name": template["provider_name"] if template else None,
        "client_reference": client_reference,
        "agent_action_id": spec["agent_action_id"],
        # Written before the provider call, so a crash between the two leaves a
        # message that is visibly unsent rather than no evidence at all.
        "delivery_status": "queued",
    }).execute().data[0]
    return row, False


def _settle_outbound(message_id: str, patch: dict) -> dict:
    return (
        get_db().table("conversation_messages")
        .update(patch).eq("id", message_id).execute().data[0]
    )


def _touch_outbound(conversation_id: str, preview: str, at: str) -> None:
    get_db().table("conversations").update({
        "last_outbound_at": at,
        "last_message_at": at,
        "last_message": preview[:200],
    }).eq("id", conversation_id).execute()


async def send(
    conversation: dict,
    *,
    body: str | None = None,
    purpose: str | None = None,
    template_values: dict | None = None,
    language: str = "en",
    author: str = "agent",
    author_name: str | None = None,
    author_member: str | None = None,
    agent_action_id: str | None = None,
    client_reference: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    image_url: str | None = None,
) -> SentMessage:
    """
    Send free text, a photograph, or an approved template when the window has
    closed.

    `body` and `purpose` are alternatives. Give `body` for a normal reply;
    give `purpose` to send a template deliberately. Giving `body` outside the
    window raises `MESSAGE_WINDOW_CLOSED` rather than quietly falling back to a
    template — the caller has to decide, because a template costs money and says
    something different from what the caller wrote.
    """
    channel = conversation.get("channel") or policy.WHATSAPP
    pol = policy.for_channel(channel)

    connection = await run_db(_connection_for_property, conversation["property_id"], channel)
    if not connection:
        raise MessagingError(
            NO_CONNECTION,
            f"property {conversation['property_id']} has no connected "
            f"{channel} account",
        )

    template = None
    if purpose:
        # A channel with no template mechanism refuses by name rather than
        # falling back to free text. The fallback would look like a success and
        # would be rejected by Meta outside the window anyway — and inside the
        # window it would send something other than what the caller asked for.
        if not pol.supports_templates:
            raise MessagingError(
                TEMPLATE_NOT_FOUND,
                f"{channel} has no approved-template mechanism; "
                f"a {purpose!r} message must be sent as free text inside the window",
            )
        template = await templates_module.resolve(
            conversation["property_id"], purpose=purpose, language=language,
            channel=channel,
            allow_test=bool(connection.get("is_sandbox")),
        )
    elif not window_open(conversation):
        closes = window_closes_at(conversation)
        # The remedy differs by channel, so the message says which one applies.
        # An agent that reads "send a template" on Instagram will try, and the
        # refusal it gets back is one turn wasted in front of a guest.
        remedy = ("an approved template may still be sent" if pol.supports_templates
                  else "there is no template mechanism on this channel")
        raise MessagingError(
            WINDOW_CLOSED,
            f"the {channel} service window closed at "
            + (closes.isoformat() if closes else "an unknown time")
            + f"; free-form messages are not permitted until the guest writes again — {remedy}",
        )

    if body is None and template is None and image_url is None:
        raise MessagingError(PROVIDER_FAILED, "nothing to send")

    # A photograph goes through this function rather than beside it, and that is
    # deliberate. `send` is the one choke point every outbound message passes:
    # the window rule, the credit gate, the idempotent queue row and the meter
    # all live here, and a parallel media path would miss all four — which is
    # exactly how a receptionist's reply and the pipeline's fallback send once
    # went unbilled.
    if image_url is not None and template is not None:
        raise MessagingError(
            PROVIDER_FAILED, "a template and a photograph are different sends",
        )

    # The credit gate on outbound, and it is deliberately narrow.
    #
    # It applies only to a *proactive* template send — a pre-arrival nudge, a
    # follow-up, a re-engagement. Those are the things §8.9 pauses at zero.
    #
    # It does not apply to:
    #   · a reply to a guest, agent or human. The window is open because
    #     somebody wrote to the hotel, and leaving them unanswered to collect a
    #     credit debt is not a decision this codebase offers.
    #   · a booking confirmation, payment receipt, cancellation or check-in
    #     link. Those draw on the overdraft — withholding a confirmation for a
    #     room a guest has already paid for is structurally impossible here,
    #     which is the point of `overdraft_limit` existing at all.
    if purpose and not billing.is_critical(purpose):
        decision = await billing.allowed(
            usage_type="whatsapp.marketing",
            property_id=conversation["property_id"],
        )
        if not decision.allowed:
            raise MessagingError(
                INSUFFICIENT_CREDITS,
                f"proactive {purpose} messages are paused: the account is out of credits "
                f"(balance {decision.balance}). Replies and confirmations still send.",
            )

    reference = client_reference or f"msg:{uuid.uuid4()}"
    message, replayed = await run_db(_queue_outbound, conversation, {
        "body": body, "template": template, "author": author,
        "author_name": author_name, "author_member": author_member,
        "agent_action_id": agent_action_id, "client_reference": reference,
        "image_url": image_url,
    })

    # Already sent. A retry after a provider timeout lands here, which is the
    # whole reason the reference is chosen before the request leaves.
    if replayed and message.get("delivery_status") in ("sent", "delivered", "read"):
        return SentMessage(
            message=message, external_id=message.get("external_id"),
            template_name=message.get("template_name"), duplicate=True,
        )

    adapter = await creds_module.adapter_for_async(connection)
    try:
        if template is not None:
            result = await adapter.send_template(
                to=conversation.get("guest_phone") or conversation.get("guest_handle") or "",
                template_name=template["provider_name"],
                language=template["language"],
                parameters=templates_module.bind(template, template_values or {}),
                client_reference=reference,
            )
        elif image_url is not None:
            result = await adapter.send_image(
                to=conversation.get("guest_phone") or conversation.get("guest_handle") or "",
                image_url=image_url,
                # Instagram ignores this and sends the words separately; WhatsApp
                # carries it on the picture. Both are correct for their channel
                # and neither is this caller's problem.
                caption=body,
                client_reference=reference,
            )
        else:
            result = await adapter.send_text(
                to=conversation.get("guest_phone") or conversation.get("guest_handle") or "",
                body=body or "",
                client_reference=reference,
            )
    except MessagingError as exc:
        # The intent survives the failure. A failed row is what makes a retry
        # possible and what stops the inbox claiming a guest was answered.
        await run_db(_settle_outbound, message["id"], {
            "delivery_status": "failed",
            "failure_reason": exc.detail[:400],
            "error_code": exc.code,
        })
        raise

    now = datetime.now(timezone.utc).isoformat()
    message = await run_db(_settle_outbound, message["id"], {
        "delivery_status": "sent",
        "external_id": result.external_id,
        "sent_at": now,
        "metadata": result.raw,
    })
    preview = body or (template or {}).get("body_preview") or ("📷 Photo" if image_url else "")
    await run_db(_touch_outbound, conversation["id"], preview, now)

    # Metered here rather than in the agent's tool, because this is the one
    # choke point every outbound message passes through — the same argument as
    # the window rule. Metering in the tool left a receptionist's reply and the
    # pipeline's own fallback send unbilled, which is the kind of gap that is
    # only ever found by reconciling against a provider invoice.
    #
    # Idempotent on the message id, so the retry path above — which returns
    # without reaching here — cannot double-count either.
    try:
        from app.services.ai import usage as usage_mod

        await usage_mod.record_message(
            message=message,
            template_name=template["provider_name"] if template else None,
            agent_id=agent_id,
            run_id=run_id,
            action_id=agent_action_id,
        )
    except Exception:  # noqa: BLE001
        logger.warning("could not meter outbound message %s", message["id"], exc_info=True)

    return SentMessage(
        message=message, external_id=result.external_id,
        template_name=template["provider_name"] if template else None,
    )


# ── delivery callbacks ───────────────────────────────────────────────────────

def _apply_status(update: DeliveryUpdate) -> dict | None:
    db = get_db()
    rows = (
        db.table("conversation_messages")
        .select("id,delivery_status,metadata,property_id,conversation_id")
        .eq("channel", update.channel)
        .eq("external_id", update.external_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        # A status for a message we never sent. Not an error worth failing the
        # webhook over — Meta will keep redelivering it — but worth a line.
        logger.info("delivery status for unknown message %s", update.external_id)
        return None

    row = rows[0]
    history = list((row.get("metadata") or {}).get("statuses") or [])
    history.append({"status": update.status, "at": (update.at or datetime.now(timezone.utc)).isoformat()})

    metadata = {**(row.get("metadata") or {}), "statuses": history[-10:]}

    # Meta's billing dimensions, kept beside the message they describe. The
    # charge itself is raised in `apply_delivery_status` below — not here,
    # because this runs on the DB executor thread and charging is async, and
    # not in the webhook handler, because a status for a message we never sent
    # must not become a charge.
    billing = {
        k: v for k, v in {
            "conversationId": update.conversation_id_external,
            "category": update.conversation_category,
            "pricingModel": update.pricing_model,
            "billable": update.billable,
        }.items() if v is not None
    }
    if billing:
        metadata["providerBilling"] = {**(metadata.get("providerBilling") or {}), **billing}

    patch = {
        "delivery_status": update.status,
        "metadata": metadata,
    }
    if update.status == "failed":
        patch["failure_reason"] = (update.error_detail or "")[:400] or None
        patch["error_code"] = update.error_code

    # The database refuses a backwards move — see guard_delivery_status in
    # migration 11 — so a late `delivered` after a `failed` updates the history
    # and leaves the status alone.
    return db.table("conversation_messages").update(patch).eq("id", row["id"]).execute().data[0]


async def apply_delivery_status(update: DeliveryUpdate) -> dict | None:
    row = await run_db(_apply_status, update)
    if row is None:
        return None

    # The conversation window is the unit Meta charges for, and the callback is
    # the only place it is ever named. Charged once per window: the key is
    # Meta's conversation id, and Meta repeats the block on several statuses.
    #
    # `billable is False` is Meta telling us this one is free — a service
    # conversation inside the free tier, or a re-delivery. Charging the hotel
    # for something we were not charged for would be indefensible; recording it
    # at a zero rate we invented would be worse.
    if update.conversation_id_external and update.billable is not False:
        try:
            from app.services.ai import usage as usage_mod

            await usage_mod.record_conversation(
                property_id=row.get("property_id"),
                channel=update.channel,
                category=update.conversation_category or "service",
                external_conversation_id=update.conversation_id_external,
                conversation_id=row.get("conversation_id"),
                message_id=row.get("id"),
                country=update.recipient_country,
                pricing_model=update.pricing_model,
            )
        except Exception:  # noqa: BLE001
            logger.warning("could not meter conversation %s",
                           update.conversation_id_external, exc_info=True)

    return row


# ── human takeover ───────────────────────────────────────────────────────────

def _set_handler(conversation_id: str, property_id: str, patch: dict) -> dict | None:
    rows = (
        get_db().table("conversations").update(patch)
        .eq("id", conversation_id).eq("property_id", property_id).execute().data
    )
    return rows[0] if rows else None


async def request_human(conversation_id: str, property_id: str, reason: str | None = None) -> dict | None:
    """
    The agent stepping back. Nobody has picked it up yet, and the inbox has to
    be able to show that gap — it is real and it lasts minutes.
    """
    patch = {"handler": "human_requested"}
    if reason in ("complaint", "refund", "policy_conflict"):
        patch["intent"] = "complaint" if reason == "complaint" else "support"
    return await run_db(_set_handler, conversation_id, property_id, patch)


async def take_over(conversation_id: str, property_id: str, member) -> dict | None:
    """A person picking it up. Stops the agent until somebody explicitly resumes."""
    return await run_db(_set_handler, conversation_id, property_id, {
        "handler": "human",
        "assignee_member": member.member_id,
        "taken_over_by": member.profile_id,
        "taken_over_at": datetime.now(timezone.utc).isoformat(),
        "unread_count": 0,
    })


async def resume_ai(conversation_id: str, property_id: str) -> dict | None:
    """
    Handing it back. Deliberately explicit: a conversation does not drift back
    to the AI because a human stopped typing.
    """
    return await run_db(_set_handler, conversation_id, property_id, {
        "handler": "ai",
        "ai_enabled": True,
        "taken_over_by": None,
        "taken_over_at": None,
    })


def agent_may_answer(conversation: dict) -> bool:
    """
    **Human takeover stops autonomous replies**, as a column rather than a
    convention. Checked before a run is opened, so a taken-over conversation
    costs nothing and leaves no trace.
    """
    return (
        conversation.get("ai_enabled", True) is not False
        and (conversation.get("handler") or "ai") == "ai"
    )


# ── a paid booking, and the guest who is waiting to hear about it ────────────
#
# Both of these run from `_fulfil_booking`, after a signed payment has produced
# a reservation, and both are best-effort by construction. A guest who has paid
# ends up with a booking whether or not we can message them about it; the
# reverse — a message claiming a booking that does not exist — is the failure
# the guardrails were written for.


def _link_booking(reservation: dict) -> dict | None:
    """
    Point the thread at the stay it produced, and at the guest who made it.

    Three writes that are really one fact: this conversation now has a booking.
    Without it `get_booking_status` has to search, `ctx.reservation_id` stays
    empty, and `list_guest_reservations` — which starts from `guest_id` —
    returns nothing to a guest who booked ninety seconds ago.

    Guarded on emptiness rather than overwritten. A thread that already names a
    reservation is a guest with two bookings, and the older one is the one they
    are more likely to be asking about; the newer is reachable through the
    guest record either way.
    """
    conversation_id = reservation.get("conversation_id")
    if not conversation_id:
        return None

    db = get_db()
    patch: dict = {"reservation_id": reservation["id"]}
    if reservation.get("guest_id"):
        patch["guest_id"] = reservation["guest_id"]
    if reservation.get("guest_name"):
        patch["guest_name"] = reservation["guest_name"]

    updated = (
        db.table("conversations").update(patch)
        .eq("id", conversation_id)
        .is_("reservation_id", "null")
        .execute().data or []
    )
    # Read rather than rely on the update's return: the guard above means a
    # thread that already names an older booking updates nothing, and that
    # guest still needs their handle recorded.
    conversation = updated[0] if updated else _conversation_by_id(conversation_id)

    # And the other direction: teach the guest record which Instagram account
    # this is, so the *next* thread they open resolves to them.
    #
    # Here rather than in `_resolve_guest`, because that function is called from
    # the booking service and has no idea a messaging channel exists — it is
    # given a name, a phone and an email, and on Instagram two of those are
    # null. This is the one moment where the handle and the guest record are
    # both in scope: somebody has just stopped being a stranger.
    handle = (conversation or {}).get("guest_handle")
    if handle and reservation.get("guest_id")             and (conversation or {}).get("channel") == "instagram":
        try:
            db.table("guests").update({"instagram_id": handle}).eq(
                "id", reservation["guest_id"]
            ).is_("instagram_id", "null").execute()
        except Exception:  # noqa: BLE001
            # A unique-index collision means this IGSID already belongs to a
            # guest row at this property — an earlier booking under a different
            # name. Leaving the older row holding the handle is right: it has
            # the stay history.
            logger.info("could not claim instagram id for guest %s",
                        reservation.get("guest_id"), exc_info=True)

    return conversation


async def link_booking(reservation: dict) -> dict | None:
    """See `_link_booking`. Swallows its own failures: the booking is real either way."""
    try:
        return await run_db(_link_booking, reservation)
    except Exception:  # noqa: BLE001
        logger.warning("could not link reservation %s to its conversation",
                       reservation.get("id"), exc_info=True)
        return None


def _confirmation_body(reservation: dict, language: str, url: str) -> str:
    """
    What a guest reads when their money lands.

    Written here, in three languages, rather than generated. This message is
    sent by a webhook with no model on the path and no guest turn to answer —
    and it is the one message in the product that must be exactly true, because
    a guest acts on it by travelling. Composing it from a template means the
    reference, the dates and the amounts come from the reservation row and
    cannot be anything else.
    """
    ref = reservation.get("ref") or ""
    check_in = str(reservation.get("check_in") or "")
    check_out = str(reservation.get("check_out") or "")
    units = int(reservation.get("units") or 1)
    paid = int(reservation.get("paid_paise") or 0)
    total = int(reservation.get("total_paise") or 0)
    balance = max(0, total - paid)

    def rupees(paise: int) -> str:
        return f"₹{paise / 100:,.0f}"

    room_word = {"en": "room" if units == 1 else "rooms",
                 "hi": "कमरा" if units == 1 else "कमरे",
                 "hinglish": "room" if units == 1 else "rooms"}[language]

    if language == "hi":
        lines = [
            "पेमेंट मिल गया ✅ आपकी बुकिंग कन्फर्म हो गई है 🎉",
            f"बुकिंग नंबर: {ref}",
            f"{check_in} से {check_out} तक, {units} {room_word}",
        ]
        if balance:
            lines.append(f"अभी दिया: {rupees(paid)} · होटल में देना है: {rupees(balance)}")
        else:
            lines.append(f"पूरा भुगतान हो गया: {rupees(paid)}")
        lines.append(f"पूरी डिटेल यहाँ देखें: {url}")
        return "\n".join(lines)

    if language == "hinglish":
        lines = [
            "Payment mil gaya ✅ Aapki booking confirm ho gayi hai 🎉",
            f"Booking number: {ref}",
            f"{check_in} se {check_out} tak, {units} {room_word}",
        ]
        if balance:
            lines.append(f"Abhi diya: {rupees(paid)} · Hotel par dena hai: {rupees(balance)}")
        else:
            lines.append(f"Poora payment ho gaya: {rupees(paid)}")
        lines.append(f"Poori detail yahan dekhein: {url}")
        return "\n".join(lines)

    lines = [
        "Payment received ✅ Your booking is confirmed 🎉",
        f"Booking reference: {ref}",
        f"{check_in} to {check_out}, {units} {room_word}",
    ]
    if balance:
        lines.append(f"Paid now: {rupees(paid)} · Due at the hotel: {rupees(balance)}")
    else:
        lines.append(f"Paid in full: {rupees(paid)}")
    lines.append(f"Your booking and receipt: {url}")
    return "\n".join(lines)


def booking_url(reservation: dict) -> str:
    """
    The link a guest is sent when their money lands.

    `/b/{public_token}` on the website, not `/checkin/{ref}` on the partner app.
    Two reasons, and both were mistakes worth naming:

      * **The reference is not a credential.** It is six characters from a
        32-symbol alphabet produced by `random.choices` — built to be read
        aloud, not to guard a page. `reservations.public_token` is 128 bits
        from `gen_random_bytes`, and rotating it revokes the old link.
      * **A confirmation is not a check-in form.** The guest who just paid wants
        the reference, the dates, the money and the cancellation terms; the
        check-in form is one link on from there, and the page carries it.

    A reservation with no token — a row read through a projection that did not
    select it — falls back to the old check-in URL rather than producing a link
    to nowhere.
    """
    token = reservation.get("public_token")
    if token:
        return f"{settings.site_url.rstrip('/')}/b/{token}"
    base = (settings.pms_app_url or "").rstrip("/")
    return f"{base}/checkin/{reservation.get('ref')}"


def _conversation_by_id(conversation_id: str) -> dict | None:
    rows = (
        get_db().table("conversations").select("*")
        .eq("id", conversation_id).limit(1).execute().data or []
    )
    return rows[0] if rows else None


async def announce_booking(reservation: dict) -> dict | None:
    """
    Tell the guest their payment landed, on the thread they paid from.

    **This is the message whose absence broke a real sale.** A guest paid at
    19:58:52; at 19:59:18 he asked whether it had gone through, and the agent —
    which had no way to know — told him it would be confirmed once the payment
    processed. It already had. Nothing in the system was going to volunteer
    that, because nothing in the system had a reason to speak unprompted.

    Idempotent on the reservation: `client_reference` is the booking, so the
    browser callback and the webhook racing to fulfil the same order produce
    one message. A window that has closed, a channel with no template, a
    provider that is down — all of them are logged and none of them raise. The
    guest can still ask, and `get_booking_status` will now have the answer.
    """
    conversation_id = reservation.get("conversation_id")
    if not conversation_id:
        return None

    conversation = await run_db(_conversation_by_id, conversation_id)
    if not conversation:
        return None

    language = language_module.of_conversation(conversation)
    body = _confirmation_body(reservation, language, booking_url(reservation))

    # The sale is over, so the session describing it is too.
    #
    # `guest_profile` carried `stage: "booked"`, a spent `quote_id` and the
    # dates of a stay that is now a row in `reservations`. Nothing cleared it,
    # so a guest writing back six weeks later about different nights started
    # inside the finished transaction — and `session.describe` put September's
    # dates into the system prompt of an October conversation. What the guest
    # is owed after this point is `get_booking_status`, which reads the
    # reservation and needs none of it.
    #
    # Before the message rather than after: the message can legitimately fail
    # (a closed window, a channel with no template) and the session is stale
    # either way.
    await run_db(session_module.close, conversation_id)

    try:
        sent = await send(
            conversation,
            body=body,
            author="system",
            author_name="Booking",
            client_reference=f"booking:{reservation['id']}",
        )
    except MessagingError as exc:
        if exc.code == WINDOW_CLOSED:
            # Outside the window free text is not allowed. On WhatsApp an
            # approved template still is, and a booking confirmation is exactly
            # what templates are for; on Instagram there is no such mechanism
            # and the guest will be told when they next write.
            pol = policy.for_channel(conversation.get("channel") or policy.WHATSAPP)
            if pol.supports_templates:
                try:
                    return (await send(
                        conversation,
                        purpose="booking_confirmation",
                        language=language if language != "hinglish" else "en",
                        author="system",
                        author_name="Booking",
                        client_reference=f"booking:{reservation['id']}",
                        template_values={
                            "ref": reservation.get("ref") or "",
                            "check_in": str(reservation.get("check_in") or ""),
                            "check_out": str(reservation.get("check_out") or ""),
                        },
                    )).message
                except MessagingError as template_exc:
                    logger.info("no booking confirmation template: %s", template_exc.code)
                    return None
        logger.info("could not announce booking %s: %s", reservation.get("ref"), exc.code)
        return None
    except Exception:  # noqa: BLE001
        logger.warning("could not announce booking %s", reservation.get("ref"), exc_info=True)
        return None

    return sent.message
