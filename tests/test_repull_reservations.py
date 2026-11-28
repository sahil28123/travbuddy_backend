"""
Channel bookings into HostOS: what a guest booking on Booking.com does to the
hotel's reservations and its stock.

The two things that cost a hotel money here are both about the counter. A
booking that never takes its room lets HostOS sell the same room again on its
own website; a cancellation that releases a room twice puts a room on sale the
hotel does not have. So every test below asserts on `room_inventory` as the
`book_inventory` / `release_booking` RPCs would leave it, not only on the row.

The RPCs are modelled (`Stock`) rather than recorded, because the questions are
"how many rooms are sold on the 10th now", and a list of calls answers that
only after arithmetic the test would then have to get right itself.
"""
import asyncio
import json
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.pms import booking
from app.services.repull import client as repull_client
from app.services.repull import reservations, signature
from tests.fake_supabase import FakeDb, install

PROPERTY = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
PARTNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
DELUXE = "22222222-2222-4222-8222-222222222222"
SUITE = "33333333-3333-4333-8333-333333333333"
DELUXE_PLAN = "11111111-1111-4111-8111-111111111111"
SUITE_PLAN = "44444444-4444-4444-8444-444444444444"
WORKSPACE = "ws_hotel"

CHECK_IN = date.today() + timedelta(days=30)


def run(coro):
    return asyncio.run(coro)


class Stock:
    """`room_inventory.units_booked` per room type per night, with a capacity."""

    def __init__(self, capacity: dict[str, int]):
        self.capacity = capacity
        self.booked: dict[tuple[str, str], int] = {}

    def _nights(self, p):
        day = date.fromisoformat(p["p_check_in"][:10])
        end = date.fromisoformat(p["p_check_out"][:10])
        while day < end:
            yield (p["p_room_type"], day.isoformat())
            day += timedelta(days=1)

    def book(self, p):
        keys = list(self._nights(p))
        if any(self.booked.get(k, 0) + p["p_units"] > self.capacity[k[0]] for k in keys):
            return -1  # refused, nothing moved
        for k in keys:
            self.booked[k] = self.booked.get(k, 0) + p["p_units"]
        return min(self.capacity[k[0]] - self.booked[k] for k in keys)

    def release(self, p):
        for k in self._nights(p):
            self.booked[k] = self.booked.get(k, 0) - p["p_units"]
        return None

    def sold(self, room: str, day: date) -> int:
        return self.booked.get((room, day.isoformat()), 0)


@pytest.fixture
def hotel(monkeypatch):
    fake = FakeDb()
    fake.unique = {"reservations": (("external_source", "external_id"),)}
    fake.seed("properties", {"id": PROPERTY, "partner_id": PARTNER, "name": "Casa Test",
                             "check_in_time": "14:00", "policies": {}})
    fake.seed("room_types",
              {"id": DELUXE, "property_id": PROPERTY, "name": "Deluxe", "total_units": 2},
              {"id": SUITE, "property_id": PROPERTY, "name": "Suite", "total_units": 1})
    fake.seed("rate_plans",
              {"id": DELUXE_PLAN, "property_id": PROPERTY, "room_type_id": DELUXE,
               "name": "Deluxe", "audience": "public", "refundable": True},
              {"id": SUITE_PLAN, "property_id": PROPERTY, "room_type_id": SUITE,
               "name": "Suite", "audience": "public", "refundable": True})
    fake.seed("repull_connections", {"property_id": PROPERTY, "workspace_id": WORKSPACE})
    fake.seed("repull_room_map",
              {"room_type_id": DELUXE, "property_id": PROPERTY, "rate_plan_id": DELUXE_PLAN,
               "repull_listing_id": "501", "unit_count": 2},
              {"room_type_id": SUITE, "property_id": PROPERTY, "rate_plan_id": SUITE_PLAN,
               "repull_listing_id": "502", "unit_count": 1})
    stock = Stock({DELUXE: 2, SUITE: 1})
    fake.rpcs["book_inventory"] = stock.book
    fake.rpcs["release_booking"] = stock.release
    install(monkeypatch, fake)

    s = repull_client.settings
    monkeypatch.setattr(s, "enable_hotel_os", True)
    monkeypatch.setattr(s, "enable_repull", True)
    monkeypatch.setattr(s, "repull_api_key", "sk_test")

    pushes, escalations = [], []

    async def push(property_id, room_type_id, from_date, to_date):
        pushes.append((room_type_id, str(from_date), str(to_date)))

    async def escalate(**kwargs):
        escalations.append(kwargs)
        return {"id": "esc", "duplicate": False, "notified": True}

    monkeypatch.setattr("app.services.repull.ari_push.enqueue_ari_push", push)
    monkeypatch.setattr("app.services.pms.escalation.raise_escalation", escalate)
    fake.stock, fake.pushes, fake.escalations = stock, pushes, escalations
    return fake


def obj(**over) -> dict:
    base = {
        "id": 9001, "listingId": 501, "platform": "booking", "status": "confirmed",
        "confirmationCode": "BK-4411", "checkIn": CHECK_IN.isoformat(),
        "checkOut": (CHECK_IN + timedelta(days=2)).isoformat(),
        "primaryGuest": {"firstName": "Asha", "lastName": "Rao",
                         "email": "asha@example.com", "phone": "+919812300000"},
        "occupancy": {"adults": 2, "children": 1},
        "totalPrice": 12345.67, "currency": "INR",
        "conversation": {"messages": ["not kept"]},
    }
    base.update(over)
    return base


def upsert(o, **kw):
    return run(reservations.upsert_external_reservation(PROPERTY, o, **kw))


def booked_rows(db):
    return [r for r in db.rows("reservations") if r.get("external_id")]


# ── a new booking ────────────────────────────────────────────────────────────

def test_a_channel_booking_becomes_a_confirmed_reservation_and_takes_its_room(hotel):
    result = upsert(obj())
    assert result["outcome"] == "created"
    [row] = booked_rows(hotel)
    assert row["status"] == "confirmed"
    assert row["source"] == "booking" and row["external_source"] == "booking"
    assert row["external_id"] == "9001" and row["external_ref"] == "BK-4411"
    assert row["guest_name"] == "Asha Rao" and row["guest_phone"] == "+919812300000"
    assert (row["adults"], row["children"], row["units"]) == (2, 1, 1)
    assert row["rate_plan_id"] == DELUXE_PLAN and row["ref"].startswith("TB-")
    # Rupees to paise, unrounded by HostOS pricing: the channel's number.
    assert row["total_paise"] == row["gross_paise"] == 1234567
    # Trimmed: the conversation is not copied onto the row.
    assert "conversation" not in row["external_payload"]
    assert row["external_payload"]["hostos"]["stockBooked"] is True
    # A guest record, linked.
    assert row["guest_id"] and hotel.rows("guests")[0]["name"] == "Asha Rao"
    # Stock: one Deluxe sold on both nights, not on check-out day.
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 1
    assert hotel.stock.sold(DELUXE, CHECK_IN + timedelta(days=1)) == 1
    assert hotel.stock.sold(DELUXE, CHECK_IN + timedelta(days=2)) == 0
    # The night trail sums back to the stay.
    nights = hotel.rows("reservation_nights")
    assert len(nights) == 2 and sum(n["room_paise"] for n in nights) == 1234567
    assert hotel.pushes == [(DELUXE, CHECK_IN.isoformat(),
                             (CHECK_IN + timedelta(days=2)).isoformat())]
    assert hotel.escalations == []


def test_a_replayed_webhook_changes_nothing(hotel):
    upsert(obj())
    calls_before = len(hotel.rpc_calls)
    again = upsert(obj())
    assert again["outcome"] == "unchanged"
    assert len(booked_rows(hotel)) == 1
    assert len(hotel.rpc_calls) == calls_before
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 1
    assert len(hotel.pushes) == 1


def test_a_non_rupee_total_is_kept_but_not_converted(hotel):
    upsert(obj(totalPrice=180.5, currency="usd"))
    [row] = booked_rows(hotel)
    assert row["total_paise"] == 0 and row["gross_paise"] == 0
    assert row["external_payload"]["totalPrice"] == 180.5
    assert row["external_payload"]["currency"] == "usd"
    assert "USD 180.50" in row["notes"]
    assert hotel.rows("reservation_nights") == []


def test_a_listing_with_no_room_type_is_ignored_not_failed(hotel):
    result = upsert(obj(listingId=999))
    assert result["outcome"] == "ignored"
    assert booked_rows(hotel) == []
    assert hotel.stock.booked == {}


def test_a_pending_request_is_stored_without_taking_stock_until_confirmed(hotel):
    upsert(obj(status="pending"))
    [row] = booked_rows(hotel)
    assert row["status"] == "pending"
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 0

    assert upsert(obj(status="confirmed"))["outcome"] == "updated"
    [row] = booked_rows(hotel)
    assert row["status"] == "confirmed"
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 1


# ── changes ──────────────────────────────────────────────────────────────────

def test_new_dates_release_the_old_nights_and_take_the_new(hotel):
    upsert(obj())
    moved_in = CHECK_IN + timedelta(days=5)
    result = upsert(obj(checkIn=moved_in.isoformat(),
                        checkOut=(moved_in + timedelta(days=3)).isoformat(),
                        totalPrice=15000))
    assert result["outcome"] == "updated"
    [row] = booked_rows(hotel)
    assert row["check_in"] == moved_in.isoformat() and row["nights"] == 3
    assert row["total_paise"] == 1500000
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 0
    assert all(hotel.stock.sold(DELUXE, moved_in + timedelta(days=i)) == 1 for i in range(3))
    nights = hotel.rows("reservation_nights")
    assert len(nights) == 3 and sum(n["room_paise"] for n in nights) == 1500000
    # Both ranges pushed: the old nights are back on sale on every channel.
    assert (DELUXE, CHECK_IN.isoformat(), (CHECK_IN + timedelta(days=2)).isoformat()) \
        in hotel.pushes[1:]
    assert len(hotel.pushes) == 3


def test_a_move_to_another_room_type_moves_the_stock(hotel):
    upsert(obj())
    upsert(obj(listingId=502))
    [row] = booked_rows(hotel)
    assert row["room_type_id"] == SUITE and row["rate_plan_id"] == SUITE_PLAN
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 0
    assert hotel.stock.sold(SUITE, CHECK_IN) == 1


def test_a_guest_change_on_the_same_dates_moves_no_stock(hotel):
    upsert(obj())
    before = len(hotel.rpc_calls)
    upsert(obj(primaryGuest={"firstName": "Asha", "lastName": "Rao-Iyer"}))
    [row] = booked_rows(hotel)
    assert row["guest_name"] == "Asha Rao-Iyer"
    assert len(hotel.rpc_calls) == before
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 1
    assert len(hotel.pushes) == 1


# ── cancellations ────────────────────────────────────────────────────────────

def test_a_cancellation_gives_the_room_back_once(hotel):
    upsert(obj())
    first = upsert(obj(status="cancelled", statusDetail="guest_cancelled"))
    assert first["outcome"] == "cancelled"
    [row] = booked_rows(hotel)
    assert row["status"] == "cancelled"
    assert row["cancellation_reason"] == "guest_cancelled"
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 0

    # The event again, as reservation.cancelled this time: still released once.
    second = upsert(obj(), cancelled=True)
    assert second["outcome"] == "unchanged"
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 0
    assert len([c for c in hotel.rpc_calls if c[0] == "release_booking"]) == 1


def test_cancelling_a_booking_hotel_os_never_stored_is_ignored(hotel):
    assert upsert(obj(status="cancelled"))["outcome"] == "ignored"
    assert booked_rows(hotel) == []


# ── overbooking ──────────────────────────────────────────────────────────────

def test_no_room_left_still_records_the_booking_and_escalates(hotel):
    upsert(obj(id=1, listingId=502))  # the only Suite
    result = upsert(obj(id=2, listingId=502, confirmationCode="BK-OVER"))
    assert result["outcome"] == "created"
    over = next(r for r in booked_rows(hotel) if r["external_id"] == "2")
    assert over["status"] == "confirmed"
    assert over["external_payload"]["hostos"]["stockBooked"] is False
    assert "Overbooked" in over["notes"]
    assert hotel.stock.sold(SUITE, CHECK_IN) == 1  # not 2: the counter never lied
    [esc] = hotel.escalations
    assert esc["kind"] == "other" and esc["reservation_id"] == over["id"]
    assert esc["facts"]["booking_ref"] == "BK-OVER"

    # A replay neither escalates again nor books anything.
    upsert(obj(id=2, listingId=502, confirmationCode="BK-OVER"))
    assert len(hotel.escalations) == 1
    assert hotel.stock.sold(SUITE, CHECK_IN) == 1

    # Cancelling the overbooked stay must not release the other guest's room.
    upsert(obj(id=2, listingId=502, status="cancelled"))
    assert hotel.stock.sold(SUITE, CHECK_IN) == 1


def test_a_move_onto_full_nights_keeps_the_change_and_escalates(hotel):
    upsert(obj(id=1, listingId=502))
    later = CHECK_IN + timedelta(days=10)
    upsert(obj(id=2, listingId=502, checkIn=later.isoformat(),
               checkOut=(later + timedelta(days=1)).isoformat()))
    # Booking 2 moves onto booking 1's nights.
    upsert(obj(id=2, listingId=502))
    row = next(r for r in booked_rows(hotel) if r["external_id"] == "2")
    assert row["check_in"] == CHECK_IN.isoformat()
    assert row["external_payload"]["hostos"]["stockBooked"] is False
    assert hotel.stock.sold(SUITE, later) == 0
    assert hotel.stock.sold(SUITE, CHECK_IN) == 1
    assert len(hotel.escalations) == 1


# ── the webhook entry point ──────────────────────────────────────────────────

def test_a_sparse_event_is_fetched_whole_from_repull(hotel, monkeypatch):
    asked = {}

    async def fake_call(fn, *, workspace_id=None, **kwargs):
        asked.update(kwargs, workspace_id=workspace_id, fn=fn.__module__)
        return obj()

    monkeypatch.setattr(reservations, "call", fake_call)
    outcome = run(reservations.handle_event(PROPERTY, {
        "event": "reservation.created", "data": {"object": {"id": 9001}},
    }))
    assert outcome == "processed"
    assert asked["id"] == 9001 and asked["workspace_id"] == WORKSPACE
    assert asked["fn"].endswith("get_reservation")
    assert len(booked_rows(hotel)) == 1


def test_a_booking_from_a_connected_pms_carries_the_pms_name(hotel):
    run(reservations.handle_event(PROPERTY, {
        "event": "reservation.created", "account": {"provider": "mews"},
        "data": {"object": obj(platform="direct")},
    }))
    [row] = booked_rows(hotel)
    assert row["external_source"] == "mews"


def test_handle_event_reports_ignored_for_an_unmapped_listing(hotel):
    outcome = run(reservations.handle_event(PROPERTY, {
        "event": "reservation.created", "data": {"object": obj(listingId=12345)},
    }))
    assert outcome == "ignored"


def test_handle_event_is_inert_when_repull_is_off(hotel, monkeypatch):
    monkeypatch.setattr(repull_client.settings, "enable_repull", False)
    assert run(reservations.handle_event(PROPERTY, {
        "event": "reservation.created", "data": {"object": obj()},
    })) == "ignored"
    assert booked_rows(hotel) == []


# ── the shared insert did not change HostOS's own booking path ──────────────

def test_create_reservation_still_books_and_compensates(hotel):
    from datetime import datetime, timezone
    quote = {
        "id": "q1", "property_id": PROPERTY, "partner_id": PARTNER, "status": "active",
        "room_type_id": DELUXE, "rate_plan_id": DELUXE_PLAN, "hold_id": None,
        "check_in": CHECK_IN.isoformat(),
        "check_out": (CHECK_IN + timedelta(days=1)).isoformat(),
        "units": 1, "adults": 2, "children": 0, "gross_paise": 500000,
        "discount_paise": 0, "tax_paise": 0, "total_paise": 500000,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "nightly_paise": [{"date": CHECK_IN.isoformat(), "price_paise": 500000,
                           "room_paise": 500000, "tax_paise": 0}],
    }
    hotel.seed("quotes", quote)
    made = run(booking.create_reservation(
        property_id=PROPERTY, quote_id="q1", guest_name="Walk In", guest_phone=None,
        guest_email=None, source="walk_in", token_paise=0, payment_method=None,
        notes=None, created_by="employee", member_id=None, member_name=None,
    ))
    assert made["ref"].startswith("TB-") and made["status"] == "confirmed"
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 1

    # A failing insert gives the room back instead of stranding it.
    hotel.seed("quotes", {**quote, "id": "q2"})
    hotel.unique["reservations"] = (("quote_id",),)
    hotel.rows("reservations")[-1]["quote_id"] = "q2"
    with pytest.raises(RuntimeError):
        run(booking.create_reservation(
            property_id=PROPERTY, quote_id="q2", guest_name="Walk In", guest_phone=None,
            guest_email=None, source="walk_in", token_paise=0, payment_method=None,
            notes=None, created_by="employee", member_id=None, member_name=None,
        ))
    assert hotel.stock.sold(DELUXE, CHECK_IN) == 1


# ── the route ────────────────────────────────────────────────────────────────

SECRET = "whsec_test"


@pytest.fixture
def webhook(monkeypatch):
    from app.api.v1.pms import repull as route

    monkeypatch.setattr(route.settings, "repull_webhook_secret", SECRET)
    seen = {"claimed": [], "marked": [], "released": [], "handled": []}

    async def property_for_workspace(ws):
        return PROPERTY if ws == WORKSPACE else None

    async def claim(event_id, event_type, workspace_id, envelope):
        seen["claimed"].append(event_id)
        return True

    async def mark(event_id, outcome, error=None):
        seen["marked"].append((event_id, outcome))

    async def release(event_id, reason):
        seen["released"].append((event_id, reason))

    async def handle(property_id, envelope):
        seen["handled"].append((property_id, envelope["event"]))
        if seen.get("explode"):
            raise RuntimeError("database went away")
        return "processed"

    monkeypatch.setattr(route.connections, "property_for_workspace", property_for_workspace)
    monkeypatch.setattr(route.events, "claim", claim)
    monkeypatch.setattr(route.events, "mark", mark)
    monkeypatch.setattr(route.events, "release", release)
    monkeypatch.setattr(reservations, "handle_event", handle)

    app = FastAPI()
    app.include_router(route.router, prefix="/api/v1")
    return TestClient(app), seen


def _post(client, body: dict, *, secret=SECRET):
    raw = json.dumps(body).encode()
    return client.post("/api/v1/pms/public/repull/webhook", content=raw, headers={
        "X-Repull-Signature": signature.sign(secret, raw),
        "X-Repull-Event-Id": body["eventId"],
        "Content-Type": "application/json",
    })


ENVELOPE = {"event": "reservation.created", "eventId": "evt_1", "workspaceId": WORKSPACE,
            "data": {"object": {"id": 9001}}}


def test_a_signed_webhook_is_handled(webhook):
    client, seen = webhook
    response = _post(client, ENVELOPE)
    assert response.status_code == 200
    assert response.json() == {"ok": True, "handled": "processed"}
    assert seen["handled"] == [(PROPERTY, "reservation.created")]
    assert seen["marked"] == [("evt_1", "processed")]


def test_a_badly_signed_webhook_is_refused(webhook):
    client, seen = webhook
    response = _post(client, ENVELOPE, secret="not-the-secret")
    assert response.status_code == 401
    assert seen["handled"] == [] and seen["claimed"] == []


def test_a_handler_failure_answers_500_and_releases_the_event(webhook):
    client, seen = webhook
    seen["explode"] = True
    response = _post(client, ENVELOPE)
    assert response.status_code == 500
    assert seen["released"] and seen["released"][0][0] == "evt_1"
    assert "database went away" in seen["released"][0][1]
    assert seen["marked"] == []
