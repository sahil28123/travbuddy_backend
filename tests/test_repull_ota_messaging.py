"""
OTA guest messages, end to end, without the network.

A guest writes on an Airbnb thread; Repull sends `reservation.message.received`;
HostOS records it as an `ota` conversation linked to the guest's booking and
hands it to the agent; the agent's reply goes back through Repull into the same
thread. Each step is asserted on what a hotel would see — the conversation row,
the message, what reached the sandbox outbox, the request Repull would have
received — not on which helper was called.

Repull itself is faked at `client.call`, the one function every request goes
through, so the SDK still builds the real request object.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.services.messaging import credentials as creds_module
from app.services.messaging import policy
from app.services.messaging import service as messaging
from app.services.messaging.channels import (
    NOT_CONFIGURED,
    PROVIDER_FAILED,
    PROVIDER_UNAVAILABLE,
    TEMPLATE_NOT_FOUND,
    MessagingError,
)
from app.services.messaging.ota import webhook as ota_webhook
from app.services.messaging.ota.adapter import OtaAdapter
from app.services.messaging.ota.fake import FakeOtaAdapter
from app.services.repull.client import RepullError
from tests.fake_supabase import FakeDb, install

PROPERTY = "p0p0p0p0-0001-4001-8001-0000000000aa"
PARTNER = "b0b0b0b0-0001-4001-8001-0000000000aa"
OTHER_PROPERTY = "p0p0p0p0-0001-4001-8001-0000000000bb"
WORKSPACE = "sandbox"
THREAD = "532457"
REPULL_RESERVATION = 215906


def event(*, message_id="1847801", body="Hi! What time can we check in?",
          role="guest", direction="inbound", workspace=WORKSPACE, **extra) -> dict:
    data = {
        "reservationId": REPULL_RESERVATION,
        "threadId": THREAD,
        "externalThreadId": "2603530383",
        "listingId": 5668,
        "channel": "airbnb",
        "messageId": message_id,
        "externalMessageId": f"328{message_id}",
        "confirmationCode": "HMEXAMPLE1",
        "from": {"type": role, "name": "Alex Morgan"},
        "body": body,
        "sentAt": "2026-11-28T09:41:15.000Z",
        "direction": direction,
        "senderType": role,
        "senderName": "Alex Morgan",
        "attachments": [],
    }
    data.update(extra)
    return {
        "event": "reservation.message.received",
        "eventId": f"evt-{message_id}",
        "apiVersion": "2026-09-01",
        "timestamp": "2026-11-28T09:41:16.000Z",
        "account": {"id": 77, "provider": "airbnb", "externalAccountId": "a1"},
        "workspaceId": workspace,
        "data": data,
    }


@pytest.fixture
def db(monkeypatch):
    fake = FakeDb()
    fake.unique = {"conversation_messages": (("channel", "external_id"),)}
    fake.seed("properties", {"id": PROPERTY, "partner_id": PARTNER,
                             "timezone": "Asia/Kolkata", "name": "Casa"})
    # partner_id deliberately null: the column is nullable on repull_connections
    # and the resolver has to find the partner through the property.
    fake.seed("repull_connections", {"property_id": PROPERTY, "partner_id": None,
                                     "workspace_id": WORKSPACE, "status": "active"})
    fake.seed("reservations", {
        "id": "res-1", "property_id": PROPERTY, "guest_id": "guest-1",
        "external_source": "airbnb", "external_id": str(REPULL_RESERVATION),
        "status": "confirmed", "check_in": "2026-12-01", "check_out": "2026-12-04",
    })
    install(monkeypatch, fake)
    FakeOtaAdapter.reset()
    yield fake
    FakeOtaAdapter.reset()


@pytest.fixture
def dispatched(monkeypatch):
    calls: list[str] = []

    async def fake_dispatch(conversation_id, **_kw):
        calls.append(conversation_id)

    monkeypatch.setattr("app.services.messaging.pipeline.dispatch", fake_dispatch)
    return calls


def handle(envelope, property_id=PROPERTY) -> str:
    return asyncio.run(ota_webhook.handle_event(property_id, envelope))


# ── in ───────────────────────────────────────────────────────────────────────

def test_a_guest_message_becomes_an_ota_conversation_on_the_booking(db, dispatched):
    assert handle(event()) == "processed"

    [conversation] = db.rows("conversations")
    assert conversation["channel"] == "ota"
    assert conversation["property_id"] == PROPERTY
    assert conversation["partner_id"] == PARTNER
    # The Repull thread is both the identity and the reply address.
    assert conversation["external_id"] == THREAD
    assert conversation["guest_handle"] == THREAD
    assert conversation["guest_phone"] is None
    assert conversation["channel_account_id"] == WORKSPACE
    # Linked through Repull's reservation id, and the guest through the booking.
    assert conversation["reservation_id"] == "res-1"
    assert conversation["guest_id"] == "guest-1"
    assert conversation["guest_name"] == "Alex Morgan"
    assert conversation["unread_count"] == 1

    [message] = db.rows("conversation_messages")
    assert message["direction"] == "inbound"
    assert message["channel"] == "ota"
    assert message["external_id"] == "1847801"
    assert message["body"] == "Hi! What time can we check in?"
    assert message["metadata"]["confirmationCode"] == "HMEXAMPLE1"

    assert dispatched == [conversation["id"]]
    assert db.rows("repull_connections")[0]["last_inbound_at"]


def test_the_route_hands_message_events_to_this_handler(db, dispatched):
    """The Repull webhook dispatcher, not just the handler, reaches the agent."""
    from app.api.v1.pms import repull as repull_routes

    outcome = asyncio.run(repull_routes._route(
        "reservation.message.received", PROPERTY, event()))
    assert outcome == "processed"
    assert len(dispatched) == 1


def test_a_second_message_lands_on_the_same_conversation(db, dispatched):
    handle(event(message_id="1"))
    handle(event(message_id="2", body="Also, is parking free?"))

    assert len(db.rows("conversations")) == 1
    assert len(db.rows("conversation_messages")) == 2
    assert db.rows("conversations")[0]["unread_count"] == 2


def test_a_photo_from_the_guest_is_kept(db, dispatched):
    handle(event(body="", attachments=[
        {"id": "a1", "url": "https://files.repull.dev/a1.jpg",
         "type": "image", "contentType": "image/jpeg"}]))

    [message] = db.rows("conversation_messages")
    assert message["media_urls"] == ["https://files.repull.dev/a1.jpg"]
    assert message["message_type"] == "image"
    assert len(dispatched) == 1, "a photo is still answered"


@pytest.mark.parametrize("role,direction", [
    ("host", "inbound"),        # the hotel's own side of the thread
    ("airbnb", "inbound"),      # Airbnb Support writing into the thread
    ("guest", "outbound"),      # an echo of something we sent
])
def test_only_the_guest_side_wakes_the_agent(db, dispatched, role, direction):
    assert handle(event(role=role, direction=direction)) == "ignored"
    assert db.rows("conversation_messages") == []
    assert db.rows("conversations") == []
    assert dispatched == []


@pytest.mark.parametrize("role", ["booker", "cotraveler"])
def test_the_booker_and_cotravellers_are_the_guest_side(db, dispatched, role):
    assert handle(event(role=role)) == "processed"
    assert len(dispatched) == 1


def test_a_replayed_message_is_recorded_and_answered_once(db, dispatched):
    """Repull retries, and a dashboard replay is a new event with the same message."""
    first = handle(event())
    replay = event()
    replay["eventId"] = "evt-replayed-from-dashboard"
    second = handle(replay)

    assert (first, second) == ("processed", "ignored")
    assert len(db.rows("conversation_messages")) == 1
    assert len(db.rows("conversations")) == 1
    assert len(dispatched) == 1


def test_a_message_for_a_disconnected_workspace_is_not_recorded(db, dispatched):
    db.rows("repull_connections")[0]["status"] = "disconnected"
    assert handle(event()) == "ignored"
    assert db.rows("conversation_messages") == []
    assert dispatched == []


def test_a_workspace_that_resolves_to_another_hotel_is_refused(db, dispatched):
    assert handle(event(), property_id=OTHER_PROPERTY) == "ignored"
    assert db.rows("conversation_messages") == []


def test_an_inquiry_with_no_booking_yet_is_still_answered(db, dispatched):
    handle(event(reservationId=None))

    [conversation] = db.rows("conversations")
    assert conversation.get("reservation_id") is None
    assert len(dispatched) == 1


def test_a_thread_id_never_matches_a_guest_by_phone(db, dispatched):
    """
    A Repull conversation id is all digits. The WhatsApp phone match would have
    paired this thread with any guest whose number ends in them.
    """
    db.seed("guests", {"id": "stranger", "property_id": PROPERTY,
                       "name": "Someone Else", "phone": "+919876532457"})
    handle(event(reservationId=None, threadId="9876532457"))
    assert db.rows("conversations")[0].get("guest_id") is None


# ── out ──────────────────────────────────────────────────────────────────────

def test_the_agent_reply_goes_back_into_the_same_thread(db, dispatched):
    handle(event())
    conversation = db.rows("conversations")[0]

    sent = asyncio.run(messaging.send(
        conversation, body="Check-in is from 2pm.", author="agent",
        client_reference="run:abc",
    ))

    last = FakeOtaAdapter.last()
    assert last is not None
    assert (last.to, last.body, last.client_reference) == (THREAD, "Check-in is from 2pm.", "run:abc")
    assert sent.message["delivery_status"] == "sent"
    assert sent.message["channel"] == "ota"


def test_an_ota_thread_is_answerable_long_after_the_guest_wrote(db, dispatched):
    """No 24-hour window: a host may answer an OTA guest at any time."""
    handle(event())
    conversation = dict(db.rows("conversations")[0])
    conversation["last_inbound_at"] = (
        datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    assert messaging.window_open(conversation) is True
    asyncio.run(messaging.send(conversation, body="Your room is ready."))
    assert FakeOtaAdapter.last().body == "Your room is ready."


def test_a_template_is_refused_on_an_ota_thread(db, dispatched):
    handle(event())
    conversation = db.rows("conversations")[0]

    with pytest.raises(MessagingError) as exc:
        asyncio.run(messaging.send(conversation, purpose="pre_arrival"))
    assert exc.value.code == TEMPLATE_NOT_FOUND
    assert FakeOtaAdapter.outbox == []


def test_a_failed_send_is_visible_as_failed(db, dispatched):
    handle(event())
    conversation = db.rows("conversations")[0]
    FakeOtaAdapter.fail_next = MessagingError(PROVIDER_UNAVAILABLE, "busy", retryable=True)

    with pytest.raises(MessagingError):
        asyncio.run(messaging.send(conversation, body="hello", client_reference="run:x"))
    outbound = [m for m in db.rows("conversation_messages") if m["direction"] == "outbound"]
    assert outbound[0]["delivery_status"] == "failed"
    assert outbound[0]["error_code"] == PROVIDER_UNAVAILABLE


def test_the_sandbox_workspace_selects_the_fake_and_a_real_one_needs_repull(monkeypatch):
    from app.config import settings

    assert isinstance(creds_module.adapter_for(
        {"channel": "ota", "workspace_id": "sandbox"}), FakeOtaAdapter)

    monkeypatch.setattr(settings, "enable_repull", False)
    with pytest.raises(MessagingError) as exc:
        creds_module.adapter_for({"channel": "ota", "workspace_id": "4021"})
    assert exc.value.code == NOT_CONFIGURED

    monkeypatch.setattr(settings, "enable_hotel_os", True)
    monkeypatch.setattr(settings, "enable_repull", True)
    monkeypatch.setattr(settings, "repull_api_key", "sk_test_x")
    adapter = creds_module.adapter_for({"channel": "ota", "workspace_id": "4021"})
    assert isinstance(adapter, OtaAdapter) and adapter.workspace_id == "4021"


# ── the adapter, against a faked Repull ─────────────────────────────────────

@pytest.fixture
def repull(monkeypatch):
    seen: dict = {}
    state = {"answer": {"id": "1854462", "status": "sent"}, "raise": None}

    async def fake_call(fn, *, workspace_id=None, **kwargs):
        seen.update(fn=fn, workspace_id=workspace_id, **kwargs)
        if state["raise"]:
            raise state["raise"]
        return state["answer"]

    monkeypatch.setattr("app.services.repull.client.call", fake_call)
    return seen, state


def test_a_text_reply_is_one_idempotent_request_in_the_hotels_workspace(repull):
    from repull.api.conversations import send_conversation_message

    seen, _ = repull
    result = asyncio.run(OtaAdapter(workspace_id="4021").send_text(
        to=THREAD, body="See you soon", client_reference="run:42"))

    assert seen["fn"] is send_conversation_message.asyncio_detailed
    assert seen["workspace_id"] == "4021"
    assert seen["id"] == int(THREAD)
    assert seen["idempotency_key"] == "run:42"
    assert seen["body"].to_dict() == {"message": "See you soon"}
    assert (result.external_id, result.status) == ("1854462", "sent")


def test_a_queued_send_is_reported_as_queued_not_retried(repull):
    _, state = repull
    state["answer"] = {"id": "1854463", "status": "queued"}
    result = asyncio.run(OtaAdapter(workspace_id="4021").send_text(
        to=THREAD, body="hi", client_reference="r"))
    assert (result.external_id, result.status) == ("1854463", "queued")


def test_a_photo_goes_as_an_attachment_with_its_caption(repull):
    seen, _ = repull
    asyncio.run(OtaAdapter(workspace_id="4021").send_image(
        to=THREAD, image_url="https://cdn.example.com/deluxe.png",
        caption="The Deluxe room", client_reference="r"))
    assert seen["body"].to_dict() == {
        "message": "The Deluxe room",
        "attachments": [{"url": "https://cdn.example.com/deluxe.png",
                         "contentType": "image/png"}],
    }


def test_the_adapter_refuses_templates():
    with pytest.raises(MessagingError) as exc:
        asyncio.run(OtaAdapter(workspace_id="4021").send_template(
            to=THREAD, template_name="x", language="en", parameters=[],
            client_reference="r"))
    assert exc.value.code == TEMPLATE_NOT_FOUND


@pytest.mark.parametrize("error,code,retryable", [
    (RepullError("rate_limit_exceeded", "slow down", status=429, retryable=True),
     PROVIDER_UNAVAILABLE, True),
    (RepullError("network_error", "ConnectTimeout", retryable=True),
     PROVIDER_UNAVAILABLE, True),
    (RepullError("account_disconnected", "Airbnb account is disconnected", status=422,
                 fix="Reconnect the Airbnb account from the channels screen."),
     PROVIDER_FAILED, False),
    (RepullError("not_configured", "REPULL_API_KEY is not set", fix="Set REPULL_API_KEY."),
     NOT_CONFIGURED, False),
])
def test_repull_failures_map_to_codes_the_agent_branches_on(repull, error, code, retryable):
    _, state = repull
    state["raise"] = error
    with pytest.raises(MessagingError) as exc:
        asyncio.run(OtaAdapter(workspace_id="4021").send_text(
            to=THREAD, body="hi", client_reference="r"))
    assert exc.value.code == code
    assert exc.value.retryable is retryable
    if error.fix:
        assert error.fix in exc.value.detail, "Repull's own remedy reaches the inbox"


def test_a_non_numeric_address_is_refused_before_repull_is_called(repull):
    seen, _ = repull
    with pytest.raises(MessagingError) as exc:
        asyncio.run(OtaAdapter(workspace_id="4021").send_text(
            to="not-a-thread", body="hi", client_reference="r"))
    assert exc.value.code == PROVIDER_FAILED
    assert "fn" not in seen


def test_ota_is_a_policy_channel():
    assert "ota" in policy.channels()
