"""
The calendar out to Repull: what a channel is told, and that telling it never
costs the hotel a save or a sale.

Two halves, tested separately because they fail in different ways:

  the push    what one night becomes (free stock, the unit-count clamp, paise
              to rupees), how nights group into calls, and what happens when
              Repull refuses part of it — including `managed_by_pms`, where the
              same nights must still go out without the unit count.

  the hooks   that a calendar edit, a booking and a cancel each queue a push,
              and that a broker outage at that moment is a log line rather
              than an error in the hotelier's face.

Repull is never called: `client.call` is replaced with a recorder. The calendar
reads are replaced too — what they return is the contract the push depends on,
and the SQL behind them is proven elsewhere.
"""
import asyncio
from datetime import date, timedelta

import pytest

from app.services.pms import ari
from app.services.repull import ari_push
from app.services.repull.client import RepullError
from tests import fake_supabase
from tests.fake_supabase import FakeDb

PROPERTY = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
ROOM_TYPE = "22222222-2222-4222-8222-222222222222"
OTHER_ROOM_TYPE = "33333333-3333-4333-8333-333333333333"
RATE_PLAN = "11111111-1111-4111-8111-111111111111"
B2B_PLAN = "44444444-4444-4444-8444-444444444444"
LISTING = "9001"
START = date(2026, 12, 1)


def run(coro):
    return asyncio.run(coro)


def inv(day: date, *, total=8, booked=0, held=0, blocked=0, stop=False, room=ROOM_TYPE) -> dict:
    return {"roomTypeId": room, "date": day.isoformat(), "unitsTotal": total,
            "unitsBooked": booked, "unitsHeld": held, "unitsBlocked": blocked,
            "unitsLeft": total - booked - held - blocked + 2,  # oversell: never used
            "stopSell": stop}


def rate(day: date, *, paise=420000, min_stay=1, cta=False, ctd=False, closed=False,
         plan=RATE_PLAN, max_stay=None) -> dict:
    return {"ratePlanId": plan, "date": day.isoformat(), "pricePaise": paise,
            "minStay": min_stay, "maxStay": max_stay, "closedToArrival": cta,
            "closedToDeparture": ctd, "closed": closed}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr("app.services.repull.client.enabled", lambda: True)


@pytest.fixture
def db(monkeypatch):
    fake = FakeDb()
    fake.seed("repull_connections", {"property_id": PROPERTY, "workspace_id": "ws_1",
                                     "status": "active"})
    fake.seed("repull_room_map", {"room_type_id": ROOM_TYPE, "property_id": PROPERTY,
                                  "rate_plan_id": RATE_PLAN, "repull_listing_id": LISTING,
                                  "unit_count": 8})
    fake_supabase.install(monkeypatch, fake)
    return fake


@pytest.fixture
def calendar(monkeypatch):
    """`nights[date] = (inventory_row, rate_row)`; the reads serve windows of it."""
    state = {"inv": {}, "rates": {}, "windows": []}

    async def get_inventory(property_id, from_date, days):
        assert days <= ari.MAX_WINDOW_DAYS
        state["windows"].append((from_date, days))
        return [r for d, r in state["inv"].items()
                if from_date <= d < from_date + timedelta(days=days)]

    async def get_rates(property_id, from_date, days):
        assert days <= ari.MAX_WINDOW_DAYS
        return [r for d, r in state["rates"].items()
                if from_date <= d < from_date + timedelta(days=days)]

    monkeypatch.setattr(ari, "get_inventory", get_inventory)
    monkeypatch.setattr(ari, "get_rates", get_rates)
    return state


@pytest.fixture
def repull(monkeypatch):
    """Records every write; `fail` is a list of callables(body) that may raise."""
    state = {"calls": [], "fail": [], "response": None}

    async def call(fn, *, workspace_id=None, **kwargs):
        body = kwargs["body"].to_dict()
        state["calls"].append({"workspace": workspace_id, "listing": kwargs["property_id"],
                               "body": body})
        for check in state["fail"]:
            check(body)
        return state["response"] or {"synced": {"channels": []}}

    monkeypatch.setattr("app.services.repull.client.call", call)
    return state


def fill(calendar, days: int, **inv_kw):
    for i in range(days):
        d = START + timedelta(days=i)
        calendar["inv"][d] = inv(d, **inv_kw)
        calendar["rates"][d] = rate(d)


# ── what one night becomes ──────────────────────────────────────────────────

def test_free_stock_subtracts_booked_held_and_blocked_never_counting_oversell():
    v = ari_push.night_values(inv(START, total=8, booked=3, held=1, blocked=2),
                              rate(START), unit_count=8)
    assert v["availableUnits"] == 2
    assert v["available"] is True


def test_units_are_clamped_to_the_listings_own_count():
    # HostOS has 12 Deluxe rooms; the Booking.com room type was set up with 5.
    # Sending 12 is a 422 from Repull.
    v = ari_push.night_values(inv(START, total=12), rate(START), unit_count=5)
    assert v["availableUnits"] == 5


def test_a_single_unit_listing_gets_no_unit_count_at_all():
    v = ari_push.night_values(inv(START, total=1), rate(START), unit_count=1)
    assert "availableUnits" not in v
    assert v["available"] is True


def test_overbooked_clamps_to_zero_and_closes_the_night():
    v = ari_push.night_values(inv(START, total=4, booked=5), rate(START), unit_count=4)
    assert v["availableUnits"] == 0
    assert v["available"] is False


@pytest.mark.parametrize("inv_kw,rate_kw", [({"stop": True}, {}), ({}, {"closed": True})])
def test_stop_sell_or_a_closed_plan_is_not_available(inv_kw, rate_kw):
    v = ari_push.night_values(inv(START, **inv_kw), rate(START, **rate_kw), unit_count=8)
    assert v["available"] is False


def test_paise_become_rupees_and_restrictions_carry_over():
    v = ari_push.night_values(inv(START), rate(START, paise=419999, min_stay=2, cta=True,
                                               max_stay=14), unit_count=8)
    assert v["price"] == 4199.99
    assert v["minNights"] == 2
    assert v["maxNights"] == 14
    assert v["closedToArrival"] is True
    assert v["closedToDeparture"] is False


def test_unknown_restrictions_are_omitted_not_guessed():
    v = ari_push.night_values(inv(START), None, unit_count=8)
    assert set(v) == {"available", "availableUnits"}


# ── grouping ────────────────────────────────────────────────────────────────

def test_identical_consecutive_nights_are_one_call_and_a_change_starts_another():
    a, b = {"available": True, "price": 100.0}, {"available": True, "price": 150.0}
    nights = [(START + timedelta(days=i), a if i < 5 or i > 6 else b) for i in range(10)]
    groups = ari_push.group_nights(nights)
    assert [len(d) for d, _ in groups] == [5, 2, 3]
    assert [v for _, v in groups] == [a, b, a]


def test_a_gap_in_the_dates_breaks_a_group():
    a = {"available": True}
    groups = ari_push.group_nights([(START, a), (START + timedelta(days=2), a)])
    assert len(groups) == 2


def test_no_group_exceeds_731_dates():
    a = {"available": True}
    nights = [(START + timedelta(days=i), a) for i in range(1000)]
    assert [len(d) for d, _ in ari_push.group_nights(nights)] == [731, 269]


# ── the push ────────────────────────────────────────────────────────────────

def test_a_push_reads_in_calendar_windows_and_sends_grouped_writes(enabled, db, calendar, repull):
    fill(calendar, 200)
    # A weekend price change in the middle.
    for i in (50, 51):
        d = START + timedelta(days=i)
        calendar["rates"][d] = rate(d, paise=550000)
    out = run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START + timedelta(days=199)))

    assert all(days <= ari.MAX_WINDOW_DAYS for _, days in calendar["windows"])
    assert len(calendar["windows"]) == 2
    assert out["nights"] == 200 and out["groups"] == 3 and out["calls"] == 3
    first, middle, last = repull["calls"]
    assert first["workspace"] == "ws_1" and first["listing"] == int(LISTING)
    assert len(first["body"]["dates"]) == 50
    assert first["body"]["price"] == 4200.0 and first["body"]["availableUnits"] == 8
    assert middle["body"]["dates"] == [(START + timedelta(days=i)).isoformat() for i in (50, 51)]
    assert middle["body"]["price"] == 5500.0
    assert len(last["body"]["dates"]) == 148


def test_only_the_mapped_room_types_nights_are_sent(enabled, db, calendar, repull):
    fill(calendar, 3)
    calendar["inv"][START] = inv(START, room=OTHER_ROOM_TYPE)  # not ours → gap
    out = run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START + timedelta(days=2)))
    assert out["nights"] == 2


def test_disabled_is_a_no_op(db, calendar, repull, monkeypatch):
    monkeypatch.setattr("app.services.repull.client.enabled", lambda: False)
    out = run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START))
    assert out["skipped"] and repull["calls"] == []


def test_an_unmapped_room_type_is_a_no_op(enabled, db, calendar, repull):
    fill(calendar, 3)
    out = run(ari_push.push_room_type(PROPERTY, OTHER_ROOM_TYPE, START, START))
    assert out["skipped"] == "room type is not mapped" and repull["calls"] == []


def test_managed_by_pms_resends_without_the_unit_count_and_stops_sending_it(
        enabled, db, calendar, repull):
    fill(calendar, 4)
    calendar["rates"][START + timedelta(days=2)] = rate(START + timedelta(days=2), paise=1)

    def refuse_units(body):
        if "availableUnits" in body:
            raise RepullError("managed_by_pms", "unit counts are owned by the PMS", status=409)

    repull["fail"].append(refuse_units)
    out = run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START + timedelta(days=3)))

    bodies = [c["body"] for c in repull["calls"]]
    # group 1: refused then resent; groups 2 and 3: sent without units first time.
    assert ["availableUnits" in b for b in bodies] == [True, False, False, False]
    assert out["errors"] == []
    assert bodies[1]["available"] is True and bodies[1]["price"] == 4200.0


def test_a_retryable_error_propagates_so_celery_retries(enabled, db, calendar, repull):
    fill(calendar, 2)

    def boom(body):
        raise RepullError("rate_limit_exceeded", "slow down", status=429, retryable=True)

    repull["fail"].append(boom)
    with pytest.raises(RepullError):
        run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START + timedelta(days=1)))


def test_a_non_retryable_refusal_is_logged_and_the_rest_still_goes(enabled, db, calendar, repull):
    fill(calendar, 4)
    calendar["rates"][START + timedelta(days=3)] = rate(START + timedelta(days=3), paise=1)
    seen = []

    def refuse_first(body):
        seen.append(1)
        if len(seen) == 1:
            raise RepullError("invalid_params", "bad", status=422, fix="do better")

    repull["fail"].append(refuse_first)
    out = run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START + timedelta(days=3)))
    assert len(repull["calls"]) == 2
    assert out["errors"][0]["code"] == "invalid_params"
    assert out["errors"][0]["fix"] == "do better"


def test_channel_failures_in_an_accepted_write_are_reported(enabled, db, calendar, repull):
    fill(calendar, 1)
    repull["response"] = {"synced": {"attempted": 2, "succeeded": 1, "failed": 1, "channels": [
        {"listingId": 9001, "channel": "airbnb", "status": "accepted", "error": None},
        {"listingId": 9001, "channel": "booking", "status": "failed", "error": "rate too low"},
    ]}}
    out = run(ari_push.push_room_type(PROPERTY, ROOM_TYPE, START, START))
    assert out["failedChannels"] == [{"from": START.isoformat(), "to": START.isoformat(),
                                      "listingId": 9001, "channel": "booking",
                                      "error": "rate too low"}]


# ── enqueueing ──────────────────────────────────────────────────────────────

@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(ari_push, "_send", lambda *a: calls.append(a))
    return calls


def test_enqueue_is_a_no_op_when_disabled(db, sent, monkeypatch):
    monkeypatch.setattr("app.services.repull.client.enabled", lambda: False)
    run(ari_push.enqueue_ari_push(PROPERTY, ROOM_TYPE, START, START))
    assert sent == []


def test_enqueue_skips_unmapped_room_types(enabled, db, sent):
    run(ari_push.enqueue_ari_push(PROPERTY, OTHER_ROOM_TYPE, START, START))
    assert sent == []


def test_enqueue_sends_iso_dates(enabled, db, sent):
    run(ari_push.enqueue_ari_push(PROPERTY, ROOM_TYPE, START, START + timedelta(days=4)))
    assert sent == [(PROPERTY, ROOM_TYPE, "2026-12-01", "2026-12-05")]


def test_a_broker_outage_is_swallowed(enabled, db, monkeypatch):
    def down(*_a):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(ari_push, "_send", down)
    run(ari_push.enqueue_ari_push(PROPERTY, ROOM_TYPE, START, START))  # no raise


def test_a_stay_pushes_its_nights_not_its_departure_morning(enabled, db, sent):
    run(ari_push.enqueue_stay(PROPERTY, ROOM_TYPE, "2026-12-01", "2026-12-04"))
    assert sent == [(PROPERTY, ROOM_TYPE, "2026-12-01", "2026-12-03")]


def test_only_the_rate_plan_the_channels_sell_is_pushed(enabled, db, sent):
    run(ari_push.enqueue_rate_plan(PROPERTY, B2B_PLAN, START, START))
    assert sent == []
    run(ari_push.enqueue_rate_plan(PROPERTY, RATE_PLAN, START, START))
    assert sent == [(PROPERTY, ROOM_TYPE, "2026-12-01", "2026-12-01")]


def test_enqueue_property_queues_every_mapped_room_type(enabled, db, sent):
    out = run(ari_push.enqueue_property(PROPERTY, days=30))
    assert out["queued"] == 1 and out["roomTypes"] == [ROOM_TYPE]
    assert sent[0][3] == (date.today() + timedelta(days=29)).isoformat()


def test_the_nightly_resync_covers_active_connections_only(enabled, db, sent):
    db.seed("repull_connections", {"property_id": "pending-hotel", "workspace_id": "ws_2",
                                   "status": "pending"})
    out = run(ari_push.resync_all())
    assert out["properties"] == 1 and out["queued"] == 1


# ── the hooks ───────────────────────────────────────────────────────────────

@pytest.fixture
def pushes(monkeypatch):
    seen = {"room": [], "plan": [], "stay": []}

    async def room(*a):
        seen["room"].append(a)

    async def plan(*a):
        seen["plan"].append(a)

    async def stay(*a):
        seen["stay"].append(a)

    monkeypatch.setattr(ari_push, "enqueue_ari_push", room)
    monkeypatch.setattr(ari_push, "enqueue_rate_plan", plan)
    monkeypatch.setattr(ari_push, "enqueue_stay", stay)
    return seen


def test_set_inventory_queues_a_push(monkeypatch, pushes):
    async def fake_run_db(fn, *a):
        return 5
    monkeypatch.setattr("app.services.pms.ari.run_db", fake_run_db)
    nights = run(ari.set_inventory(PROPERTY, room_type_id=ROOM_TYPE, from_date=START,
                                   to_date=START + timedelta(days=4), units_total=6,
                                   units_blocked=None, stop_sell=None, oversell=None))
    assert nights == 5
    assert pushes["room"] == [(PROPERTY, ROOM_TYPE, START, START + timedelta(days=4))]


def test_set_rates_queues_a_push_for_its_plan(monkeypatch, pushes):
    async def fake_run_db(fn, *a):
        return 3
    monkeypatch.setattr("app.services.pms.ari.run_db", fake_run_db)
    run(ari.set_rates(PROPERTY, rate_plan_id=RATE_PLAN, from_date=START, to_date=START,
                      price_paise=500000, min_stay=None, max_stay=None,
                      closed_to_arrival=None, closed_to_departure=None, closed=None))
    assert pushes["plan"] == [(PROPERTY, RATE_PLAN, START, START)]


def test_a_failed_calendar_write_queues_nothing(monkeypatch, pushes):
    from fastapi import HTTPException

    async def fake_run_db(fn, *a):
        raise HTTPException(409, "already booked")
    monkeypatch.setattr("app.services.pms.ari.run_db", fake_run_db)
    with pytest.raises(HTTPException):
        run(ari.set_inventory(PROPERTY, room_type_id=ROOM_TYPE, from_date=START,
                              to_date=START, units_total=0, units_blocked=None,
                              stop_sell=None, oversell=None))
    assert pushes["room"] == []


def test_a_booking_queues_a_push_for_its_stay(monkeypatch, pushes):
    from app.services.pms import booking

    reservation = {"id": "r1", "room_type_id": ROOM_TYPE,
                   "check_in": "2026-12-01", "check_out": "2026-12-03"}

    async def fake_run_db(fn, *a):
        return reservation
    monkeypatch.setattr("app.services.pms.booking.run_db", fake_run_db)
    out = run(booking.create_reservation(
        property_id=PROPERTY, quote_id="q1", guest_name="Asha", guest_phone=None,
        guest_email=None, source="direct", token_paise=0, payment_method=None,
        notes=None, created_by="employee", member_id=None, member_name=None))
    assert out is reservation
    assert pushes["stay"] == [(PROPERTY, ROOM_TYPE, "2026-12-01", "2026-12-03")]


def test_a_cancel_that_released_rooms_queues_a_push(monkeypatch, pushes):
    from datetime import datetime, timezone

    from app.services.pms import cancellation
    from tests.test_cancellation import (
        PROPERTY as C_PROPERTY, RESERVATION, ROOM_TYPE as C_ROOM, make_db,
    )

    make_db(monkeypatch)
    out = run(cancellation.cancel(C_PROPERTY, RESERVATION,
                                  at=datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)))
    assert out["inventoryReleased"] is True
    assert pushes["stay"] == [(C_PROPERTY, C_ROOM, "2026-09-10", "2026-09-12")]


def test_a_cancel_that_released_nothing_queues_nothing(monkeypatch, pushes):
    from datetime import datetime, timezone

    from app.services.pms import cancellation
    from tests.test_cancellation import PROPERTY as C_PROPERTY, RESERVATION, make_db

    make_db(monkeypatch, status="pending")
    out = run(cancellation.cancel(C_PROPERTY, RESERVATION,
                                  at=datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)))
    assert out["inventoryReleased"] is False
    assert pushes["stay"] == []


def test_hooks_never_fail_the_write_when_repull_is_unreachable(enabled, db, monkeypatch):
    """The real enqueue path, with the broker down, under a real set_inventory call."""
    def down(*_a):
        raise ConnectionError("redis is down")
    monkeypatch.setattr(ari_push, "_send", down)
    db.rpcs["set_inventory"] = 1
    nights = run(ari.set_inventory(PROPERTY, room_type_id=ROOM_TYPE, from_date=START,
                                   to_date=START, units_total=6, units_blocked=None,
                                   stop_sell=None, oversell=None))
    assert nights == 1


# ── celery wiring ───────────────────────────────────────────────────────────

def test_the_push_task_is_registered_on_the_hotelos_queue_and_on_the_stayos_beat():
    from app.workers import celery_app as celery_module
    from app.workers import repull_tasks  # noqa: F401 — registers the tasks

    app = celery_module.celery_app
    assert "app.workers.repull_tasks" in app.conf.include
    for name in ("travbuddy.pms.repull_push", "travbuddy.pms.repull_resync_all"):
        assert name in app.tasks
        route = app.amqp.router.route({}, name)
        assert getattr(route["queue"], "name", route["queue"]) == "hotelos"
    stayos = {e["task"] for e in celery_module._STAYOS_SCHEDULE.values()}
    assert "travbuddy.pms.repull_resync_all" in stayos
    marketplace = {e["task"] for e in celery_module._MARKETPLACE_SCHEDULE.values()}
    assert "travbuddy.pms.repull_resync_all" not in marketplace
