"""
The parts of the Repull integration everything else stands on.

    signature   a webhook body is trusted only if Repull signed exactly it, recently
    events      a Repull retry of a failed event is handled, a finished one is not
    client      a Repull error keeps its code, its `fix`, and whether to retry
    mapping     a room type is mapped to a listing only when the match is certain

No network: Repull is faked at `client.call` / `connections.listings`.
"""
import asyncio
import json

import pytest

from app.services.repull import client, connections, events, signature
from app.services.repull.client import RepullError
from tests.fake_supabase import FakeDb

SECRET = "whsec_" + "ab" * 48
BODY = json.dumps({"event": "reservation.created", "eventId": "evt_1"}).encode()


def run(coro):
    return asyncio.run(coro)


async def straight(fn, *args, **kwargs):
    return fn(*args, **kwargs)


# ── signature ────────────────────────────────────────────────────────────────

def test_a_signed_body_is_accepted():
    signature.verify(SECRET, BODY, signature.sign(SECRET, BODY))


def test_a_changed_body_is_refused():
    header = signature.sign(SECRET, BODY)
    with pytest.raises(signature.SignatureError):
        signature.verify(SECRET, BODY.replace(b"created", b"cancelled"), header)


def test_a_replayed_old_delivery_is_refused():
    header = signature.sign(SECRET, BODY, timestamp=1_700_000_000)
    with pytest.raises(signature.SignatureError):
        signature.verify(SECRET, BODY, header, now=1_700_000_000 + 301)


def test_clock_skew_inside_the_window_is_tolerated():
    header = signature.sign(SECRET, BODY, timestamp=1_700_000_000)
    signature.verify(SECRET, BODY, header, now=1_700_000_000 - 120)


@pytest.mark.parametrize("header", [None, "", "t=abc,v1=00", "v1=00", "t=1700000000"])
def test_a_missing_or_malformed_header_is_refused(header):
    with pytest.raises(signature.SignatureError):
        signature.verify(SECRET, BODY, header, now=1_700_000_000)


def test_no_secret_configured_refuses_everything():
    with pytest.raises(signature.SignatureError):
        signature.verify(None, BODY, signature.sign(SECRET, BODY))


def test_a_wrong_secret_is_refused():
    with pytest.raises(signature.SignatureError):
        signature.verify("whsec_other", BODY, signature.sign(SECRET, BODY))


def test_any_matching_v1_passes_during_secret_rotation():
    good = signature.sign(SECRET, BODY, timestamp=1_700_000_000).split("v1=")[1]
    header = f"t=1700000000,v1={'0' * 64},v1={good}"
    signature.verify(SECRET, BODY, header, now=1_700_000_000)


# ── events ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db(monkeypatch):
    fake = FakeDb()
    fake.unique = {"repull_webhook_events": (("event_id",),)}
    monkeypatch.setattr("app.services.repull.events.get_db", lambda: fake)
    monkeypatch.setattr("app.services.repull.events.run_db", straight)
    return fake


def test_an_event_is_handled_once(db):
    assert run(events.claim("evt_1", "reservation.created", "ws_1", {})) is True
    run(events.mark("evt_1", "processed"))
    assert run(events.claim("evt_1", "reservation.created", "ws_1", {})) is False
    assert len(db.rows("repull_webhook_events")) == 1


def test_a_failed_event_is_handed_to_repulls_retry(db):
    assert run(events.claim("evt_2", "reservation.created", "ws_1", {})) is True
    run(events.release("evt_2", "booking insert failed"))
    assert run(events.claim("evt_2", "reservation.created", "ws_1", {})) is True
    # …and only one retry gets it.
    assert run(events.claim("evt_2", "reservation.created", "ws_1", {})) is False


def test_an_event_without_an_id_is_still_handled(db):
    assert run(events.claim(None, "reservation.created", "ws_1", {})) is True


# ── client ───────────────────────────────────────────────────────────────────

def test_a_repull_error_envelope_is_kept_whole():
    err = client._error_from(422, json.dumps({"error": {
        "code": "invalid_params", "message": "availableUnits can't be 9",
        "fix": "Send a number from 0 to 4.", "docs_url": "https://repull.dev/docs/errors/invalid_params",
    }}).encode())
    assert (err.code, err.status, err.retryable) == ("invalid_params", 422, False)
    assert err.fix == "Send a number from 0 to 4."
    assert "Send a number" in str(err)


@pytest.mark.parametrize("status_code,code,retryable", [
    (429, "rate_limit_exceeded", True),
    (503, "upstream_unavailable", True),
    (409, "idempotency_key_in_use", True),
    (409, "managed_by_pms", False),
    (401, "unauthorized", False),
])
def test_retryable_is_decided_by_status_and_code(status_code, code, retryable):
    err = client._error_from(status_code, json.dumps({"error": {"code": code, "message": "x"}}).encode())
    assert err.retryable is retryable


def test_a_non_json_error_still_becomes_a_repull_error():
    err = client._error_from(502, b"<html>Bad gateway</html>")
    assert err.code == "http_502" and err.retryable is True


def test_call_returns_the_json_body_and_raises_on_failure(monkeypatch):
    monkeypatch.setattr(client.settings, "repull_api_key", "sk_live_test")

    class R:
        def __init__(self, code, content):
            self.status_code, self.content, self.headers = code, content, {}

    async def ok(*, client, **kw):
        assert client._headers["X-Workspace-Id"] == "ws_9"
        assert client.token == "sk_live_test"
        return R(200, b'{"data": [1, 2]}')

    async def bad(*, client, **kw):
        return R(404, b'{"error": {"code": "not_found", "message": "no such listing"}}')

    assert run(client.call(ok, workspace_id="ws_9")) == {"data": [1, 2]}
    with pytest.raises(RepullError) as info:
        run(client.call(bad))
    assert info.value.code == "not_found"


def test_without_a_key_nothing_is_called(monkeypatch):
    monkeypatch.setattr(client.settings, "repull_api_key", None)
    with pytest.raises(RepullError) as info:
        client.client()
    assert info.value.code == "not_configured"


def test_enabled_needs_both_flags_and_a_key(monkeypatch):
    s = client.settings
    monkeypatch.setattr(s, "enable_hotel_os", True)
    monkeypatch.setattr(s, "enable_repull", True)
    monkeypatch.setattr(s, "repull_api_key", "sk_live_test")
    assert client.enabled() is True
    monkeypatch.setattr(s, "enable_repull", False)
    assert client.enabled() is False
    monkeypatch.setattr(s, "enable_repull", True)
    monkeypatch.setattr(s, "repull_api_key", None)
    assert client.enabled() is False


# ── mapping ──────────────────────────────────────────────────────────────────

PROPERTY = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def hotel(monkeypatch):
    fake = FakeDb()
    fake.seed("repull_connections", {"property_id": PROPERTY, "workspace_id": "ws_1"})
    fake.seed("room_types",
              {"id": "rt_deluxe", "property_id": PROPERTY, "name": "Deluxe Double", "total_units": 6},
              {"id": "rt_suite", "property_id": PROPERTY, "name": "Suite", "total_units": 2},
              {"id": "rt_dorm", "property_id": PROPERTY, "name": "Dorm Bed", "total_units": 10})
    fake.seed("rate_plans",
              {"id": "rp_b2b", "property_id": PROPERTY, "room_type_id": "rt_deluxe", "audience": "b2b"},
              {"id": "rp_public", "property_id": PROPERTY, "room_type_id": "rt_deluxe", "audience": "public"},
              {"id": "rp_suite", "property_id": PROPERTY, "room_type_id": "rt_suite", "audience": None})
    monkeypatch.setattr("app.services.repull.connections.get_db", lambda: fake)
    monkeypatch.setattr("app.services.repull.connections.run_db", straight)

    async def listings(property_id):
        return [
            {"id": 501, "name": "DELUXE double", "unitCount": 6},
            {"id": 502, "name": "Suite", "unitCount": 2},
            {"id": 503, "name": "Suite", "unitCount": 1},  # a second "Suite": ambiguous
            {"id": 504, "name": "Penthouse", "unitCount": 1},
        ]

    monkeypatch.setattr(connections, "listings", listings)
    return fake


def test_automap_maps_only_certain_matches(hotel):
    result = run(connections.automap(PROPERTY))
    mapped = {m["room_type_id"]: m for m in result["mapped"]}
    assert set(mapped) == {"rt_deluxe"}
    assert mapped["rt_deluxe"]["repull_listing_id"] == "501"
    assert mapped["rt_deluxe"]["unit_count"] == 6
    # The public plan is the one sold on the channels, not the B2B one.
    assert mapped["rt_deluxe"]["rate_plan_id"] == "rp_public"
    unmatched = {u["roomTypeId"]: u["candidates"] for u in result["unmatched"]}
    assert unmatched == {"rt_suite": 2, "rt_dorm": 0}


def test_a_manual_map_is_checked_against_the_workspace(hotel):
    saved = run(connections.save_map(PROPERTY, [{"roomTypeId": "rt_suite", "listingId": 502}]))
    assert saved[0]["repull_listing_id"] == "502" and saved[0]["rate_plan_id"] == "rp_suite"
    with pytest.raises(Exception) as info:
        run(connections.save_map(PROPERTY, [{"roomTypeId": "rt_suite", "listingId": 999}]))
    assert getattr(info.value, "status_code", None) == 422
    with pytest.raises(Exception) as info:
        run(connections.save_map(PROPERTY, [{"roomTypeId": "rt_other_hotel", "listingId": 502}]))
    assert getattr(info.value, "status_code", None) == 422


def test_a_webhook_finds_its_hotel_by_workspace(hotel):
    assert run(connections.property_for_workspace("ws_1")) == PROPERTY
    assert run(connections.property_for_workspace("ws_unknown")) is None
    assert run(connections.property_for_workspace(None)) is None
