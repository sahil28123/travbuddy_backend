"""
Resilience layer: per-provider concurrency cap + hard timeout on the circuit
breaker, and the global generation queue. These are the launch load-shedders.
"""
import asyncio

import pytest

from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError


def test_breaker_caps_concurrency():
    """A max_concurrency=2 breaker must never run more than 2 calls at once, even
    when 6 are launched together — the surplus queues on the semaphore."""
    br = CircuitBreaker("t-conc", max_concurrency=2, timeout=5.0)

    async def slow():
        await asyncio.sleep(0.05)
        return "ok"

    async def run():
        await asyncio.gather(*[br.call(slow, fallback=None) for _ in range(6)])

    asyncio.run(run())
    assert br.stats()["peak_in_flight"] == 2
    assert br.stats()["successes"] == 6


def test_breaker_timeout_returns_fallback_and_counts():
    """A call slower than the breaker timeout returns the fallback (never hangs)
    and is recorded as a timeout failure."""
    br = CircuitBreaker("t-timeout", failure_threshold=99, max_concurrency=4, timeout=0.05)

    async def too_slow():
        await asyncio.sleep(1.0)
        return "never"

    async def run():
        return await br.call(too_slow, fallback="fb")

    result = asyncio.run(run())
    assert result == "fb"
    s = br.stats()
    assert s["timeouts"] == 1 and s["failures"] == 1


def test_breaker_opens_and_short_circuits():
    """After threshold failures the breaker opens and later calls skip straight to
    fallback (short-circuited), without invoking the function."""
    br = CircuitBreaker("t-open", failure_threshold=2, recovery_timeout=999, max_concurrency=4, timeout=1.0)
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise RuntimeError("x")

    async def run():
        await br.call(boom, fallback=None)   # fail 1
        await br.call(boom, fallback=None)   # fail 2 -> opens
        await br.call(boom, fallback=None)   # short-circuited, boom not called

    asyncio.run(run())
    assert br.is_open
    assert calls["n"] == 2                     # third never ran
    assert br.stats()["short_circuited"] == 1


def test_guard_raises_on_open_and_records():
    """guard() raises CircuitOpenError when open (for callers like Vertex that
    handle their own errors), and records success/failure while re-raising."""
    br = CircuitBreaker("t-guard", failure_threshold=1, recovery_timeout=999, max_concurrency=2, timeout=5.0)

    async def run_fail():
        try:
            async with br.guard():
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    async def run_open():
        with pytest.raises(CircuitOpenError):
            async with br.guard():
                pass  # never reached

    asyncio.run(run_fail())   # 1 failure -> opens
    assert br.is_open
    asyncio.run(run_open())
    assert br.stats()["short_circuited"] == 1


@pytest.mark.parametrize("use_guard", [False, True], ids=["call", "guard"])
def test_breaker_rechecks_open_state_after_queueing(use_guard):
    """Calls already waiting for a slot must skip the provider if earlier
    failures open the breaker before they acquire the semaphore."""
    br = CircuitBreaker("t-queued", failure_threshold=2, recovery_timeout=999,
                        max_concurrency=1, timeout=5.0)
    calls = {"n": 0}

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def boom():
            calls["n"] += 1
            started.set()
            await release.wait()
            raise RuntimeError("provider down")

        async def invoke():
            if use_guard:
                async with br.guard():
                    return await boom()
            return await br.call(boom, fallback="fb")

        first = asyncio.create_task(invoke())
        await asyncio.wait_for(started.wait(), timeout=5.0)
        queued = [asyncio.create_task(invoke()) for _ in range(5)]
        # Let every caller reach the semaphore while the first attempt is held.
        await asyncio.sleep(0)
        release.set()
        return await asyncio.wait_for(
            asyncio.gather(first, *queued, return_exceptions=True), timeout=5.0,
        )

    results = asyncio.run(run())
    assert calls["n"] == 2
    if use_guard:
        assert sum(isinstance(result, RuntimeError) for result in results) == 2
        assert sum(isinstance(result, CircuitOpenError) for result in results) == 4
    else:
        assert results == ["fb"] * 6
    s = br.stats()
    assert s["state"] == "open"
    assert s["calls"] == 6
    assert s["failures"] == 2 and s["consecutive_failures"] == 2
    assert s["short_circuited"] == 4
    assert s["successes"] == 0 and s["timeouts"] == 0
    assert s["in_flight"] == 0 and s["peak_in_flight"] == 1


def test_generation_queue_caps_and_queues(monkeypatch):
    """The generation queue admits at most `capacity` at once; the rest wait and
    are counted as queued."""
    import app.core.generation_queue as gq

    # Fresh, small-capacity queue for the test.
    monkeypatch.setattr(gq, "_capacity", 2)
    monkeypatch.setattr(gq, "_sem", None)
    monkeypatch.setattr(gq, "_in_flight", 0)
    monkeypatch.setattr(gq, "_waiting", 0)
    monkeypatch.setattr(gq, "_peak_in_flight", 0)
    monkeypatch.setattr(gq, "_total_queued", 0)
    monkeypatch.setattr(gq, "_total_admitted", 0)

    async def work(i, seen_queued):
        async with gq.generation_slot(f"trip-{i}") as queued:
            seen_queued.append(queued)
            await asyncio.sleep(0.05)

    async def run():
        seen = []
        await asyncio.gather(*[work(i, seen) for i in range(5)])
        return seen

    asyncio.run(run())
    assert gq.stats()["peak_in_flight"] == 2      # never exceeded capacity
    assert gq.stats()["total_admitted"] == 5
    assert gq.stats()["total_queued"] >= 3        # 5 launched, 2 slots -> >=3 waited
