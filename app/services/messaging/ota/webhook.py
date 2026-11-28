"""
OTA guest messages, in.

Repull sends `reservation.message.received` when a guest writes on an Airbnb,
Booking.com, Vrbo or tiket.com thread. This turns that event into the same
`InboundMessage` a WhatsApp or Instagram webhook produces and hands it to the
same three calls, in the same order:

    resolve_connection  ──►  record_inbound  ──►  pipeline.dispatch
    (workspace → hotel)      (persist, dedup)     (the agent answers)

**The order is the safety property, as it is for Meta.** Persist before the
agent runs, so Repull's retry of a slow delivery finds the message already
recorded — `record_inbound` is idempotent on `(channel, external_id)` — and
produces no second conversation, no second run and no second reply. The
webhook route has its own event-id dedupe in front of this; the message-level
one is what holds when the *same message* arrives under two events (a
backfill, a replay from the Repull dashboard).

**Only the guest's side wakes the agent.** The event fires for anyone writing
on the guest's side of the thread — the guest, the booker, a co-traveller —
and also for Airbnb Support writing into it (`from.type: "airbnb"`). Support is
not a guest asking a question, and an agent answering it as one would be
replying to the platform as if it were a customer. Messages the host side sent
arrive as a different event (`reservation.message.sent`) and are not routed
here at all; one that somehow did would be refused by the same check.

**The payload, flat.** Unlike reservation events, the message is `data`
itself, not `data.object`: `threadId` (the Repull conversation id), `messageId`,
`externalMessageId`, `reservationId`, `from{type,name}`, `body`, `sentAt`,
`attachments[]{url,contentType,type}`. Every key is read defensively: a field
Repull adds tomorrow must not break today's parser, and one it omits must not
cost the guest an answer.
"""
import logging
from datetime import datetime

from app.services import trace
from app.services.messaging import pipeline
from app.services.messaging import service as messaging
from app.services.messaging.channels import InboundMessage, MessagingError
from app.services.messaging.policy import OTA

logger = logging.getLogger(__name__)

MESSAGE_RECEIVED = "reservation.message.received"

# Who counts as the guest. `from.type` on the event; `senderType` is the same
# role as stored, and is read when `from` is missing.
GUEST_SIDE = frozenset({"guest", "booker", "cotraveler"})

# What survives into `conversation_messages.metadata`. Enough to trace a bad
# delivery back to Repull and the OTA three days later; not the body twice.
_KEPT = ("threadId", "externalThreadId", "messageId", "externalMessageId",
         "reservationId", "confirmationCode", "listingId", "hotelId", "channel",
         "senderType", "revision")


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _role(data: dict) -> str | None:
    sender = data.get("from") if isinstance(data.get("from"), dict) else {}
    role = sender.get("type") or data.get("senderType")
    return str(role).lower() if role else None


def _attachments(data: dict) -> tuple[str, tuple[str, ...]]:
    """
    What kind of thing arrived, and where it lives.

    Repull copies every file to durable storage, so the `url` keeps working
    after the OTA's own link expires — which is what lets a receptionist open a
    guest's photograph of a broken tap the next morning.
    """
    atts = [a for a in (data.get("attachments") or []) if isinstance(a, dict)]
    urls = tuple(u for u in (a.get("url") or a.get("imageUrl") for a in atts) if u)
    if not urls:
        return "text", ()
    kinds = {str(a.get("type")) for a in atts if a.get("type")}
    kind = next(iter(kinds)) if len(kinds) == 1 else ("mixed" if kinds else "file")
    return kind, urls


def parse(envelope: dict) -> InboundMessage | None:
    """
    One Repull event → one guest message, or None for anything that is not one.

    None rather than raising: an event that is not a guest message is not a
    failure, and raising would make Repull retry it forever.
    """
    if (envelope.get("event") or MESSAGE_RECEIVED) != MESSAGE_RECEIVED:
        return None
    data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
    # Tolerated, not expected: an older delivery wrapped the message the way
    # reservation events wrap theirs.
    if "threadId" not in data and isinstance(data.get("object"), dict):
        data = data["object"]

    if (data.get("direction") or "inbound") != "inbound":
        return None
    role = _role(data)
    # A missing role on an inbound message is the guest: the event's own
    # contract is that it fires for the guest's side of the thread.
    if role is not None and role not in GUEST_SIDE:
        return None

    thread_id = data.get("threadId")
    message_id = data.get("messageId") or (
        f"{data.get('channel') or 'ota'}:{data['externalMessageId']}"
        if data.get("externalMessageId") else None
    )
    workspace_id = envelope.get("workspaceId")
    if not thread_id or not message_id or workspace_id in (None, ""):
        logger.warning("ota message event without thread, message or workspace id: %s",
                       envelope.get("eventId"))
        return None

    message_type, media = _attachments(data)
    sender = data.get("from") if isinstance(data.get("from"), dict) else {}
    raw = {k: data[k] for k in _KEPT if data.get(k) is not None}
    raw["from"] = sender or None
    raw["eventId"] = envelope.get("eventId")

    return InboundMessage(
        channel=OTA,
        # Repull's message id: unique across every OTA it carries, where the
        # OTA's own id is unique only within that OTA.
        external_id=str(message_id),
        # The conversation is the address. The OTA hides the guest's phone and
        # email, and the Repull thread is where the reply has to go anyway.
        sender_external_id=str(thread_id),
        account_id=str(workspace_id),
        message_type=message_type,
        body=(data.get("body") or None),
        media_urls=media,
        sender_name=data.get("senderName") or sender.get("name"),
        sent_at=_parse_time(data.get("sentAt")),
        raw=raw,
    )


async def handle_event(property_id: str, envelope: dict) -> str:
    """
    `reservation.message.received` → recorded and answered. Returns
    `processed` or `ignored`; raises on a failure Repull should retry.

    `property_id` is what the route resolved from `workspaceId`. The messaging
    resolver answers the same question through the connection row, and the two
    are required to agree — a disagreement means the mapping changed under us,
    and the safe answer is not to put the message into either hotel's inbox.
    """
    inbound = parse(envelope)
    if inbound is None:
        return "ignored"

    try:
        connection = await messaging.resolve_connection(inbound.account_id, OTA)
    except MessagingError as exc:
        # A workspace whose connection is disconnected. Acknowledged: a retry
        # will not reconnect it.
        logger.warning("ota message for an unconnected workspace: %s", exc.detail)
        return "ignored"
    if str(connection.get("property_id")) != str(property_id):
        logger.warning("ota workspace %s resolves to %s, not %s; not recording",
                       inbound.account_id, connection.get("property_id"), property_id)
        return "ignored"

    recorded = await messaging.record_inbound(inbound, connection)
    if recorded.duplicate:
        # The whole reason persistence comes first. No second run, no second
        # reply.
        return "ignored"

    conversation = recorded.conversation
    trace.bind(conversation_id=conversation["id"],
               hotel_id=conversation["property_id"],
               guest_id=conversation.get("guest_id"))
    logger.info("inbound ota message accepted on conversation %s", conversation["id"])

    # Enqueued after the write is durable. A worker that starts late, twice or
    # never leaves the guest visible in the inbox rather than lost.
    await pipeline.dispatch(conversation["id"])
    return "processed"
