"""
A hotel arriving from another PMS: room types matched or created once, future
bookings imported once, and a second run that changes nothing.

Repull is faked at `pms_import.call` and `connections.listings`; the database
is `FakeDb`, so the room types and rate plans are written by onboarding's own
functions exactly as the hotelier's setup screen would write them.
"""
import asyncio
from datetime import date, timedelta

import pytest

from app.services.repull import client as repull_client
from app.services.repull import connections, pms_import
from tests.fake_supabase import FakeDb, install

PROPERTY = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
PARTNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
DELUXE = "22222222-2222-4222-8222-222222222222"
DELUXE_PLAN = "11111111-1111-4111-8111-111111111111"
WORKSPACE = "ws_import"
SOON = date.today() + timedelta(days=12)


def run(coro):
    return asyncio.run(coro)


LISTINGS = [
    # Matches the room type the hotel already set up, by normalised name.
    {"id": "701", "name": "deluxe!", "unitCount": 6, "hotel": {"id": "h1"}},
    # New: priced from the listing in rupees.
    {"id": "702", "name": "Garden Suite", "unitCount": 3, "hotel": {"id": "h1"},
     "pricing": {"currency": "INR", "basePrice": 7500}},
    # New: no listing price, so priced from its own rupee bookings.
    {"id": "703", "name": "Family Room", "unitCount": 2, "hotel": {"id": "h1"}},
    # New: no rupee price anywhere — created and mapped, bookings skipped.
    {"id": "704", "name": "Loft", "unitCount": 1, "hotel": {"id": "h1"}},
]


def booking(rid, listing, nights=2, total=10000, currency="INR", status="confirmed"):
    return {"id": rid, "listingId": int(listing), "platform": "direct", "status": status,
            "checkIn": SOON.isoformat(),
            "checkOut": (SOON + timedelta(days=nights)).isoformat(),
            "primaryGuest": {"firstName": f"Guest{rid}"}, "occupancy": {"adults": 2},
            "totalPrice": total, "currency": currency, "confirmationCode": f"M-{rid}"}


BOOKINGS = {
    "701": [booking(1, 701), booking(2, 701, status="cancelled")],
    "702": [booking(3, 702, total=15000)],
    "703": [booking(4, 703, nights=2, total=9000), booking(5, 703, nights=1, total=5000)],
    "704": [booking(6, 704, total=200, currency="EUR")],
}


@pytest.fixture
def hotel(monkeypatch):
    fake = FakeDb()
    fake.unique = {"reservations": (("external_source", "external_id"),)}
    fake.seed("properties", {"id": PROPERTY, "partner_id": PARTNER, "name": "Casa Import"})
    fake.seed("room_types", {"id": DELUXE, "property_id": PROPERTY, "name": "Deluxe",
                             "code": "DELUXE", "total_units": 6})
    fake.seed("rate_plans", {"id": DELUXE_PLAN, "property_id": PROPERTY,
                             "room_type_id": DELUXE, "name": "Deluxe", "audience": "public",
                             "code": "DELUXE-EP"})
    fake.seed("repull_connections", {"property_id": PROPERTY, "workspace_id": WORKSPACE,
                                     "connected_providers": [{"provider": "mews"}]})
    fake.rpcs["book_inventory"] = lambda p: 1
    fake.rpcs["release_booking"] = lambda p: None
    install(monkeypatch, fake)

    s = repull_client.settings
    monkeypatch.setattr(s, "enable_hotel_os", True)
    monkeypatch.setattr(s, "enable_repull", True)
    monkeypatch.setattr(s, "repull_api_key", "sk_test")

    asked = {"listings_include": [], "reservations": []}

    async def listings(property_id, *, include=None):
        asked["listings_include"].append(include)
        return [dict(item) for item in LISTINGS]

    async def fake_call(fn, *, workspace_id=None, **kwargs):
        assert workspace_id == WORKSPACE
        if fn.__module__.endswith("list_hotels"):
            return {"data": [{"id": "h1", "provider": "Mews"}]}
        if fn.__module__.endswith("list_reservations"):
            asked["reservations"].append(kwargs)
            assert kwargs["check_out_after"] == date.today()
            rows = BOOKINGS[str(kwargs["listing_id"])]
            # Two pages for one listing, to prove the cursor is followed.
            if len(rows) > 1 and "cursor" not in kwargs:
                return {"data": rows[:1], "pagination": {"hasMore": True, "nextCursor": "c2"}}
            if "cursor" in kwargs:
                return {"data": rows[1:], "pagination": {"hasMore": False}}
            return {"data": rows, "pagination": {"hasMore": False}}
        raise AssertionError(f"unexpected Repull call {fn.__module__}")

    pushes = []

    async def push(property_id, room_type_id, from_date, to_date):
        pushes.append(room_type_id)

    monkeypatch.setattr(connections, "listings", listings)
    monkeypatch.setattr(pms_import, "call", fake_call)
    monkeypatch.setattr("app.services.repull.ari_push.enqueue_ari_push", push)
    fake.asked, fake.pushes = asked, pushes
    return fake


def test_import_matches_creates_and_imports_once(hotel):
    counts = run(pms_import.run_import(PROPERTY))
    assert counts == {"roomTypesCreated": 3, "roomTypesMatched": 1,
                      # 1, 3, 4, 5 recorded; 2 cancelled and never stored, 6 has
                      # no rupee rate plan to hang on.
                      "reservationsImported": 4, "reservationsSkipped": 2}
    assert hotel.asked["listings_include"] == ["pricing"]

    rooms = {r["name"]: r for r in hotel.rows("room_types")}
    assert set(rooms) == {"Deluxe", "Garden Suite", "Family Room", "Loft"}
    assert rooms["Garden Suite"]["total_units"] == 3

    plans = {p["room_type_id"]: p for p in hotel.rows("rate_plans")}
    assert plans[rooms["Garden Suite"]["id"]]["base_price_paise"] == 750000
    assert plans[rooms["Garden Suite"]["id"]]["audience"] == "public"
    # Median nightly of 4,500 and 5,000 rupees.
    assert plans[rooms["Family Room"]["id"]]["base_price_paise"] == 475000
    assert rooms["Loft"]["id"] not in plans

    room_map = {m["repull_listing_id"]: m for m in hotel.rows("repull_room_map")}
    assert room_map["701"]["room_type_id"] == DELUXE
    assert room_map["701"]["rate_plan_id"] == DELUXE_PLAN
    assert room_map["704"]["rate_plan_id"] is None
    assert room_map["702"]["unit_count"] == 3

    stays = [r for r in hotel.rows("reservations") if r.get("external_id")]
    assert sorted(r["external_id"] for r in stays) == ["1", "3", "4", "5"]
    assert {r["external_source"] for r in stays} == {"mews"}
    assert {r["status"] for r in stays} == {"confirmed"}
    assert set(hotel.pushes) == {DELUXE, rooms["Garden Suite"]["id"],
                                 rooms["Family Room"]["id"]}


def test_running_the_import_again_creates_nothing(hotel):
    run(pms_import.run_import(PROPERTY))
    snapshot = {t: len(hotel.rows(t)) for t in
                ("room_types", "rate_plans", "repull_room_map", "reservations",
                 "reservation_nights", "guests")}
    again = run(pms_import.run_import(PROPERTY))
    assert again["roomTypesCreated"] == 0
    assert again["roomTypesMatched"] == 4
    assert again["reservationsImported"] == 0
    assert again["reservationsSkipped"] == 6
    assert {t: len(hotel.rows(t)) for t in snapshot} == snapshot


def test_import_needs_a_connection(hotel):
    from fastapi import HTTPException

    hotel.store["repull_connections"] = []
    with pytest.raises(HTTPException) as exc:
        run(pms_import.run_import(PROPERTY))
    assert exc.value.status_code == 409
